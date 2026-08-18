from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


class FailureBoundary(str, Enum):
    BEFORE_OPTIMIZER_STEP = "before_optimizer_step"
    AFTER_OPTIMIZER_BEFORE_COMMIT = "after_optimizer_before_commit"
    AFTER_COMMIT = "after_commit"


@dataclass
class FailureCoordinator:
    failures: list[dict[str, str]] = field(default_factory=list)

    def record(
        self,
        *,
        attempt_id: int,
        boundary: FailureBoundary,
        component: str,
        detail: str,
    ) -> None:
        self.failures.append(
            {
                "attempt_id": str(attempt_id),
                "boundary": boundary.value,
                "component": str(component),
                "detail": str(detail),
            }
        )
        if boundary is FailureBoundary.AFTER_OPTIMIZER_BEFORE_COMMIT:
            raise RuntimeError(
                "Actor/optimizer state is untrusted after a post-step pre-commit "
                "failure. Terminate and restore the last successful checkpoint."
            )
