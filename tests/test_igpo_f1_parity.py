import json
import math

import pytest

from agentic_rl.outcome.token_f1 import compute_f1


@pytest.mark.parametrize(
    ("solution", "truth", "source", "val_type", "expected"),
    [
        ("<answer>Paris</answer>", "Paris", "", "f1", 1.0),
        ("<answer>New York</answer>", "New Jersey", "", "f1", 0.5),
        ("<answer>wrong</answer>", "Paris", "", "f1", 0.0),
        ("<answer>New-York</answer>", "new york", "", "f1", 1.0),
        ("<answer>PARIS</answer>", "paris", "", "f1", 1.0),
        ("<answer>new new york</answer>", "new york", "", "f1", 1.0),
        ("<answer>The Hague</answer>", "Hague", "", "f1", 2.0 / 3.0),
        (
            "<answer>Paris</answer>",
            "Lutetia<|answer_split|>Paris",
            "",
            "f1",
            1.0,
        ),
        ("<answer></answer>", "", "", "f1", 0.0),
        ("<answer></answer>", "Paris", "", "f1", 0.0),
        ("<answer>Paris", "Paris", "", "f1", -2.0),
        ("missing", "Paris", "", "f1", -2.0),
        (
            "<answer>Paris</answer><answer>London</answer>",
            "Paris",
            "",
            "f1",
            1.0,
        ),
        ("<answer>New\nYork</answer>", "New York", "", "f1", 1.0),
        (
            "<answer>false</answer>",
            json.dumps([{"label": "true"}, {"label": "false"}]),
            "Factbench",
            "f1",
            1.0,
        ),
    ],
)
def test_frozen_official_igpo_fixture_parity(
    solution,
    truth,
    source,
    val_type,
    expected,
) -> None:
    actual = compute_f1(solution, truth, source, val_type)
    assert math.isclose(actual, expected, abs_tol=1.0e-12)
