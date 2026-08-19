"""Explicit configuration compositions used by CPU-only unit tests."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# configs/base.yaml is deliberately abstract and has no implicit machine
# profile. Tests that need a fully resolved legacy-shaped config must opt into
# an explicit fixture instead.
TEST_CONFIG = ROOT / "tests" / "fixtures" / "base_5x48gb.yaml"
MICA_CONFIG = ROOT / "tests" / "fixtures" / "formal_train_mica_4x48gb.yaml"
PAPER_MICA_CONFIG = ROOT / "tests" / "fixtures" / "formal_train_paper_mica_4x48gb.yaml"
PILOT_CONFIG = ROOT / "tests" / "fixtures" / "pilot_20_5x48gb.yaml"
FORMAL_RESUME_CONFIG = ROOT / "tests" / "fixtures" / "formal_resume_u20_to_u500_5x48gb.yaml"
FORMAL_CONFIG = ROOT / "tests" / "fixtures" / "formal_train_5x48gb.yaml"
THREE_RANK_PARENT_CONFIG = ROOT / "tests" / "fixtures" / "formal_resume_u20_3rank_4x48gb.yaml"
