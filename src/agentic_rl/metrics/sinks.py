from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .schema import MetricScope


class JsonlMetricSink:
    def __init__(self, paths_by_scope: Mapping[MetricScope, str | Path]) -> None:
        self.paths = {
            scope: Path(path) for scope, path in paths_by_scope.items()
        }
        self._lock = threading.Lock()

    def write(self, scope: MetricScope, record: Mapping[str, Any]) -> None:
        self.write_many(scope, (record,))

    def write_many(
        self,
        scope: MetricScope,
        records: Sequence[Mapping[str, Any]],
    ) -> None:
        if scope not in self.paths:
            raise KeyError(f"No metric sink configured for {scope.value}")
        encoded = []
        for record in records:
            payload = dict(record)
            payload["scope"] = scope.value
            now = time.time()
            payload.setdefault("timestamp_unix", now)
            payload.setdefault(
                "timestamp_utc",
                datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
            )
            encoded.append(
                json.dumps(payload, sort_keys=True, ensure_ascii=True) + "\n"
            )
        if not encoded:
            return
        path = self.paths[scope]
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.writelines(encoded)
                handle.flush()
                os.fsync(handle.fileno())
