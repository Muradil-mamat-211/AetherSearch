from __future__ import annotations

import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINER_PATH = PROJECT_ROOT / "dpo" / "scripts" / "train_dpo.py"
SPEC = importlib.util.spec_from_file_location("aethersearch_dpo_trainer", TRAINER_PATH)
assert SPEC is not None and SPEC.loader is not None
TRAINER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TRAINER
SPEC.loader.exec_module(TRAINER)


class CharacterTokenizer:
    """Small reversible tokenizer for testing exact mask boundaries."""

    eos_token = TRAINER.IM_END
    eos_token_id = 0x110001
    pad_token_id = 0
    is_fast = True
    backend_tokenizer = SimpleNamespace(normalizer=None)

    @staticmethod
    def _encode(text: str) -> list[int]:
        result: list[int] = []
        cursor = 0
        while cursor < len(text):
            if text.startswith(TRAINER.IM_END, cursor):
                result.append(CharacterTokenizer.eos_token_id)
                cursor += len(TRAINER.IM_END)
            else:
                result.append(ord(text[cursor]) + 1)
                cursor += 1
        return result

    def __call__(self, text, **_kwargs):
        if isinstance(text, list):
            return SimpleNamespace(input_ids=[self._encode(item) for item in text])
        return SimpleNamespace(input_ids=self._encode(text))

    def decode(self, ids, **_kwargs) -> str:
        pieces: list[str] = []
        for token_id in ids:
            if token_id == self.eos_token_id:
                pieces.append(TRAINER.IM_END)
            else:
                pieces.append(chr(token_id - 1))
        return "".join(pieces)


def canonical_prompt(question: str) -> str:
    return (
        f"{TRAINER.SYSTEM_MARKER}System instructions.{TRAINER.IM_END}\n"
        f"{TRAINER.USER_MARKER}Answer with search. Question: {question}"
        f"{TRAINER.IM_END}\n{TRAINER.ASSISTANT_MARKER}"
    )


def canonical_record() -> dict:
    question = "Who wrote the example?"
    return {
        "id": "000001",
        "question": question,
        "source_dataset": "sample_source",
        "answers": ["Ada"],
        "pair_type": "answer_hard_negative",
        "prompt_text": canonical_prompt(question),
        "chosen": "<think>The evidence is sufficient.</think><answer>Ada</answer>",
        "rejected": "<think>I should search broadly.</think><search>example</search>",
    }


def test_answer_gets_eot_and_search_does_not(tmp_path: Path) -> None:
    record = canonical_record()
    train_file = tmp_path / "train.jsonl"
    train_file.write_text(json.dumps(record) + "\n", encoding="utf-8")
    tokenizer = CharacterTokenizer()
    dataset = TRAINER.AetherSearchDPODataset(
        train_file=str(train_file),
        tokenizer=tokenizer,
        expected_num_samples=1,
        expected_sha256=None,
        tokenization_batch_size=1,
    )

    sample = dataset[0]
    assert sample["chosen_input_ids"][-1] == tokenizer.eos_token_id
    assert sample["chosen_labels"][-1] == tokenizer.eos_token_id
    assert sample["rejected_input_ids"][-1] != tokenizer.eos_token_id
    assert sample["rejected_labels"][-1] != TRAINER.IGNORE_INDEX
    prompt_tokens = len(tokenizer._encode(record["prompt_text"]))
    assert sample["chosen_labels"][:prompt_tokens] == [
        TRAINER.IGNORE_INDEX
    ] * prompt_tokens
    assert sample["rejected_labels"][:prompt_tokens] == [
        TRAINER.IGNORE_INDEX
    ] * prompt_tokens


def test_information_is_masked_on_both_sides(tmp_path: Path) -> None:
    record = canonical_record()
    record["pair_type"] = "true_full_trajectory_preference"
    record["chosen"] = (
        "<think>Search first.</think><search>author</search>\n"
        "<information>Chosen evidence.</information>\n"
        "<think>Now answer.</think><answer>Ada</answer>"
    )
    record["rejected"] = (
        "<think>Search first.</think><search>writer</search>\n"
        "<information>Rejected evidence.</information>\n"
        "<think>Guess.</think><answer>Grace</answer>"
    )
    train_file = tmp_path / "train.jsonl"
    train_file.write_text(json.dumps(record) + "\n", encoding="utf-8")
    tokenizer = CharacterTokenizer()
    dataset = TRAINER.AetherSearchDPODataset(
        train_file=str(train_file),
        tokenizer=tokenizer,
        expected_num_samples=1,
        expected_sha256=None,
        tokenization_batch_size=1,
    )

    sample = dataset[0]
    for side in ("chosen", "rejected"):
        ids = sample[f"{side}_input_ids"]
        labels = sample[f"{side}_labels"]
        decoded = tokenizer.decode(ids)
        match = re.search(r"<information>.*?</information>", decoded)
        assert match is not None
        prefix_length = len(tokenizer._encode(decoded[: match.start()]))
        span_length = len(tokenizer._encode(match.group(0)))
        assert labels[prefix_length : prefix_length + span_length] == [
            TRAINER.IGNORE_INDEX
        ] * span_length
        assert labels[-1] == tokenizer.eos_token_id
    assert dataset.audit.information_masked_tokens["chosen"]["sum"] > 0
    assert dataset.audit.information_masked_tokens["rejected"]["sum"] > 0


def test_collator_masks_padding(tmp_path: Path) -> None:
    first = canonical_record()
    second = canonical_record()
    second["id"] = "000002"
    second["question"] = "Who wrote the longer example?"
    second["prompt_text"] = canonical_prompt(second["question"])
    second["chosen"] = (
        "<think>The evidence is sufficient and unambiguous.</think>"
        "<answer>Ada Lovelace</answer>"
    )
    train_file = tmp_path / "train.jsonl"
    train_file.write_text(
        "\n".join(json.dumps(row) for row in (first, second)) + "\n",
        encoding="utf-8",
    )
    tokenizer = CharacterTokenizer()
    dataset = TRAINER.AetherSearchDPODataset(
        train_file=str(train_file),
        tokenizer=tokenizer,
        expected_num_samples=2,
        expected_sha256=None,
        tokenization_batch_size=2,
    )
    batch = TRAINER.DPOCollator(tokenizer)([dataset[0], dataset[1]])
    for side in ("chosen", "rejected"):
        assert batch[f"{side}_input_ids"].shape == batch[f"{side}_labels"].shape
        assert batch[f"{side}_input_ids"].shape == batch[
            f"{side}_attention_mask"
        ].shape
        for row, sample in enumerate((dataset[0], dataset[1])):
            length = len(sample[f"{side}_input_ids"])
            assert torch.all(batch[f"{side}_attention_mask"][row, length:] == 0)
            assert torch.all(
                batch[f"{side}_labels"][row, length:] == TRAINER.IGNORE_INDEX
            )


def test_dpo_objective_matches_closed_form() -> None:
    policy_chosen = torch.tensor([-3.0, -2.0])
    policy_rejected = torch.tensor([-4.0, -4.0])
    reference_chosen = torch.tensor([-3.0, -2.0])
    reference_rejected = torch.tensor([-4.0, -4.0])
    losses, chosen_rewards, rejected_rewards = TRAINER.dpo_objective(
        policy_chosen,
        policy_rejected,
        reference_chosen,
        reference_rejected,
        beta=0.1,
    )
    assert torch.allclose(losses, torch.full_like(losses, math.log(2.0)))
    assert torch.equal(chosen_rewards, torch.zeros_like(chosen_rewards))
    assert torch.equal(rejected_rewards, torch.zeros_like(rejected_rewards))

    improved_losses, _, _ = TRAINER.dpo_objective(
        policy_chosen + 1.0,
        policy_rejected,
        reference_chosen,
        reference_rejected,
        beta=0.1,
    )
    assert torch.all(improved_losses < losses)


class TinyCausalModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 32, hidden_size: int = 12):
        super().__init__()
        self.config = SimpleNamespace(use_cache=True)
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.output = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        del attention_mask, use_cache
        return SimpleNamespace(logits=self.output(self.embedding(input_ids)))


def test_trainer_freezes_reference_and_computes_finite_loss(tmp_path: Path) -> None:
    torch.manual_seed(7)
    model = TinyCausalModel()
    reference = TinyCausalModel()
    reference.load_state_dict(model.state_dict())
    args = TRAINER.TrainingArguments(
        output_dir=str(tmp_path / "output"),
        per_device_train_batch_size=1,
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = TRAINER.AetherSearchDPOTrainer(
        model=model,
        reference_model=reference,
        args=args,
        beta=0.1,
        forward_mode="concatenated",
        data_collator=lambda rows: rows,
    )
    batch = {
        "chosen_input_ids": torch.tensor([[1, 2, 3, 4]]),
        "chosen_attention_mask": torch.tensor([[1, 1, 1, 1]]),
        "chosen_labels": torch.tensor([[-100, -100, 3, 4]]),
        "rejected_input_ids": torch.tensor([[1, 2, 5, 6]]),
        "rejected_attention_mask": torch.tensor([[1, 1, 1, 1]]),
        "rejected_labels": torch.tensor([[-100, -100, 5, 6]]),
    }
    loss = trainer.compute_loss(model, batch)
    assert torch.isfinite(loss)
    assert torch.allclose(loss, torch.tensor(math.log(2.0)), atol=1e-6)
    assert trainer.reference_model.training is False
    assert all(not parameter.requires_grad for parameter in reference.parameters())


def test_structural_corruption_is_rejected() -> None:
    record = canonical_record()
    record["chosen"] = "<think>Reason.</think> stray <answer>Ada</answer>"
    with pytest.raises(ValueError, match="between chosen spans"):
        TRAINER.validate_public_record(record, line_number=3, strict_public_schema=True)


def test_public_defaults_lock_canonical_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(TRAINER_PATH),
            "--model_name_or_path",
            "muradil211/AetherSearch_SFT",
            "--ref_model_name_or_path",
            "muradil211/AetherSearch_SFT",
            "--train_file",
            "train.jsonl",
            "--output_dir",
            "outputs/dpo",
        ],
    )
    args = TRAINER.parse_args()
    assert args.expected_num_samples == TRAINER.CANONICAL_PAIR_COUNT
    assert args.expected_sha256 == TRAINER.CANONICAL_DATA_SHA256
    assert args.long_sample_policy == "error"
    assert args.beta == 0.1
    assert args.learning_rate == 5e-7
    assert args.num_train_epochs == 1.0


def test_launcher_separates_recipe_from_machine_topology() -> None:
    launcher = (
        PROJECT_ROOT / "dpo" / "scripts" / "run_train_dpo_zero3.sh"
    ).read_text(encoding="utf-8")
    assert "canonical_num_samples=2126" in launcher
    assert f'canonical_data_sha256="{TRAINER.CANONICAL_DATA_SHA256}"' in launcher
    assert 'global_batch_size="${GLOBAL_BATCH_SIZE:-12}"' in launcher
    assert 'nproc_per_node="${NPROC_PER_NODE:-${visible_gpu_count}}"' in launcher
    assert "gradient_accumulation=$((global_batch_size / micro_batch_across_workers))" in launcher
    assert "--standalone" in launcher
    assert "NNODES" not in launcher
    assert "NODE_RANK" not in launcher
    assert not re.search(r"(?:export\s+)?CUDA_VISIBLE_DEVICES=", launcher)
    assert "NCCL_IB_DISABLE=" not in launcher
    assert "PYTORCH_CUDA_ALLOC_CONF=" not in launcher
    assert "OMP_NUM_THREADS=" not in launcher
    assert "/root/" not in launcher
