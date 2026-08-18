from agentic_rl.controller.prompt_sampler import ImmutableDatasetPromptSampler


def test_sampler_is_deterministic_unique_and_resumable() -> None:
    sampler = ImmutableDatasetPromptSampler(dataset_size=100, shuffle_seed=7)
    first = sampler.allocate(64)
    assert len(first) == len(set(first)) == 64
    state = sampler.state()
    restored = ImmutableDatasetPromptSampler.restore(state)
    assert restored.allocate(32) == sampler.allocate(32)
