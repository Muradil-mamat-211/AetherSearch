from agentic_rl.controller.dataset_view import (
    DeterministicNQHotpotLogicalView,
    _canonical_ground_truth,
)


SOURCE = "/root/autodl-tmp/search-r1-workspace/data/nq_hotpotqa_train/train.parquet"
IDENTITY = "b8bc8792a85e1172e52ceb5eaefb9c6065aa9c0dabf5fe4cb6004ddc4281710e"


def test_logical_training_view_is_exactly_150745_and_stable() -> None:
    view = DeterministicNQHotpotLogicalView(
        SOURCE,
        expected_identity_sha256=IDENTITY,
    )
    assert len(view) == 150745
    assert view.identity.nq_rows == 60298
    assert view.identity.hotpotqa_rows == 90447
    assert view.identity.ordered_view_identity_sha256 == IDENTITY


def test_logical_training_view_returns_globally_stable_prompt_ids() -> None:
    view = DeterministicNQHotpotLogicalView(
        SOURCE,
        expected_identity_sha256=IDENTITY,
    )
    row = view.row(0)
    assert row["prompt_global_id"].count(":") >= 2
    assert row["data_source"] in {"nq", "hotpotqa"}
    assert row["prompt_messages"]
    assert row["gold_aliases"]
    assert row["canonical_answer"]


def test_exact_ig_canonical_field_uses_scalar_or_first_ordered_answer() -> None:
    assert _canonical_ground_truth(
        {"reward_model": {"ground_truth": {"target": "scalar target"}}}
    ) == "scalar target"
    assert _canonical_ground_truth(
        {"golden_answers": ["first alias", "second alias"]}
    ) == "first alias"
