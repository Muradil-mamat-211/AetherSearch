from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Callable, Sequence

from .state_schema import ChannelCheckpointState, CheckpointMetadata


def _advise_dontneed(fd: int) -> bool:
    """Drop clean file pages after hashing/fsyncing large checkpoint files.

    This is an I/O-cache hint only.  It does not modify file contents and is
    deliberately best-effort because some filesystems do not expose
    ``posix_fadvise``.
    """

    posix_fadvise = getattr(os, "posix_fadvise", None)
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
    if posix_fadvise is None or dontneed is None:
        return False
    try:
        posix_fadvise(fd, 0, 0, dontneed)
    except OSError:
        return False
    return True


def release_file_cache(path: str | Path) -> dict[str, int | bool]:
    """Best-effort release of clean page cache below ``path``.

    Checkpoint files are immutable after commit, so releasing their clean
    pages is safe and prevents repeated integrity scans from accumulating
    tens of GiB in the cgroup's file-backed memory.  The return value is
    telemetry for tests and runtime reports.
    """

    target = Path(path)
    files = [target] if target.is_file() else sorted(
        value for value in target.rglob("*") if value.is_file()
    ) if target.is_dir() else []
    released = 0
    bytes_seen = 0
    for item in files:
        try:
            size = item.stat().st_size
            with item.open("rb") as handle:
                bytes_seen += size
                if _advise_dontneed(handle.fileno()):
                    released += 1
        except (FileNotFoundError, PermissionError, OSError):
            continue
    return {
        "files": int(len(files)),
        "released_files": int(released),
        "bytes_seen": int(bytes_seen),
        "supported": bool(released or not files),
    }


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
                _advise_dontneed(handle.fileno())
    directory_fd = os.open(str(root), os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
        _advise_dontneed(handle.fileno())
    return digest.hexdigest()


class AtomicCheckpointCommitter:
    def __init__(self, checkpoint_root: str | Path) -> None:
        self.root = Path(checkpoint_root)

    def commit(
        self,
        metadata: CheckpointMetadata,
        *,
        write_distributed_state: Callable[[Path], None],
        rank: int,
        barrier: Callable[[], None],
        gather_errors: Callable[[str | None], Sequence[str | None]],
        directory_name: str | None = None,
    ) -> Path:
        metadata.validate()
        name = (
            str(directory_name)
            if directory_name is not None
            else f"checkpoint-{metadata.successful_update_step:06d}"
        )
        if (
            not name
            or name in {".", ".."}
            or "/" in name
            or "\0" in name
        ):
            raise ValueError(f"Unsafe checkpoint directory name: {name!r}")
        temporary = self.root / f"{name}.tmp"
        final = self.root / name
        setup_error: str | None = None
        if rank == 0:
            try:
                self.root.mkdir(parents=True, exist_ok=True)
                if temporary.exists() or final.exists():
                    raise FileExistsError(
                        f"Checkpoint destination already exists: {name}"
                    )
                temporary.mkdir()
            except BaseException as exc:
                setup_error = repr(exc)
        setup_errors = [error for error in gather_errors(setup_error) if error]
        if setup_errors:
            raise RuntimeError(
                "Checkpoint setup failed: " + " | ".join(setup_errors)
            )
        barrier()
        write_error: str | None = None
        try:
            write_distributed_state(temporary)
        except BaseException as exc:
            write_error = repr(exc)
        write_errors = [error for error in gather_errors(write_error) if error]
        if write_errors:
            if rank == 0 and temporary.exists():
                shutil.rmtree(temporary)
            raise RuntimeError(
                "Distributed checkpoint write failed: " + " | ".join(write_errors)
            )
        barrier()

        finalize_error: str | None = None
        if rank == 0:
            try:
                required = ("actor", "optimizer", "scheduler")
                missing = [
                    entry for entry in required if not (temporary / entry).exists()
                ]
                if missing:
                    raise RuntimeError(
                        "Distributed checkpoint writer omitted: "
                        + ", ".join(missing)
                    )
                metadata_path = temporary / "metadata.json"
                metadata_path.write_text(
                    json.dumps(metadata.as_dict(), indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                integrity = {
                    str(path.relative_to(temporary)): _sha256(path)
                    for path in sorted(temporary.rglob("*"))
                    if path.is_file()
                }
                (temporary / "integrity.sha256.json").write_text(
                    json.dumps(integrity, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                _fsync_tree(temporary)
                os.replace(temporary, final)
                root_fd = os.open(str(self.root), os.O_RDONLY)
                try:
                    os.fsync(root_fd)
                finally:
                    os.close(root_fd)
                self.validate(final)
                latest_tmp = self.root / "latest_checkpoint.json.tmp"
                latest = self.root / "latest_checkpoint.json"
                latest_tmp.write_text(
                    json.dumps(
                        {
                            "checkpoint": final.name,
                            "successful_update_step": metadata.successful_update_step,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                with latest_tmp.open("rb") as handle:
                    os.fsync(handle.fileno())
                os.replace(latest_tmp, latest)
            except BaseException as exc:
                finalize_error = repr(exc)
        finalize_errors = [
            error for error in gather_errors(finalize_error) if error
        ]
        if finalize_errors:
            if rank == 0 and temporary.exists():
                shutil.rmtree(temporary)
            raise RuntimeError(
                "Checkpoint finalize failed: " + " | ".join(finalize_errors)
            )
        barrier()
        return final

    def validate(self, checkpoint: str | Path) -> CheckpointMetadata:
        checkpoint_path = Path(checkpoint)
        integrity_path = checkpoint_path / "integrity.sha256.json"
        metadata_path = checkpoint_path / "metadata.json"
        if not integrity_path.is_file() or not metadata_path.is_file():
            raise RuntimeError("Checkpoint metadata/integrity files are missing")
        integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
        for relative, expected in integrity.items():
            path = checkpoint_path / relative
            if not path.is_file() or _sha256(path) != expected:
                raise RuntimeError(f"Checkpoint integrity mismatch: {relative}")
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        payload["ig_channel"] = ChannelCheckpointState(**payload["ig_channel"])
        payload["outcome_channel"] = ChannelCheckpointState(
            **payload["outcome_channel"]
        )
        metadata = CheckpointMetadata(**payload)
        metadata.validate()
        return metadata
