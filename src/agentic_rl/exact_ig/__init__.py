"""Vectorized structural-attention Exact-IG task construction and scoring."""

from .task_builder import (
    ExactIGTask,
    ExactIGTaskBuilder,
    PrefixScoreSpan,
    SequentialExactIGTask,
    VectorizedExactIGTask,
)
from .target_schema import (
    ANSWER_SCAFFOLD_TEXT,
    EXACT_IG_VERSION,
    assert_exact_ig_checkpoint_compatible,
    select_canonical_answer,
)
from .sequential_oracle import (
    OracleTokenScore,
    SequentialOracleResult,
    sequential_teacher_forced_oracle,
)
from .vectorized_scorer import (
    ExactIGResult,
    VectorizedExactIGScorer,
    pack_exact_ig_microbatches,
)

__all__ = [
    "ANSWER_SCAFFOLD_TEXT",
    "EXACT_IG_VERSION",
    "assert_exact_ig_checkpoint_compatible",
    "ExactIGResult",
    "ExactIGTask",
    "ExactIGTaskBuilder",
    "OracleTokenScore",
    "PrefixScoreSpan",
    "SequentialOracleResult",
    "SequentialExactIGTask",
    "VectorizedExactIGScorer",
    "VectorizedExactIGTask",
    "pack_exact_ig_microbatches",
    "select_canonical_answer",
    "sequential_teacher_forced_oracle",
]
