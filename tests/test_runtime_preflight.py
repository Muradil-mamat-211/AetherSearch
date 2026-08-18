import subprocess

from agentic_rl.workers.ray_actors import _run_probe


def test_runtime_probe_timeout_fails_closed_with_structured_result(monkeypatch) -> None:
    def raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=["python", "-c", "probe"], timeout=180)

    monkeypatch.setattr(subprocess, "run", raise_timeout)
    result = _run_probe("python", "print('unused')")
    assert result == {
        "returncode": -1,
        "stderr": "Compatibility probe timed out after 180 seconds",
        "timeout": True,
    }
