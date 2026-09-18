from __future__ import annotations

import argparse

import pytest

from ml.training.create_piano_curriculum_audio import parse_ratios, ratio_label


def test_parse_ratios_deduplicates_and_sorts_descending() -> None:
    assert parse_ratios("0.25,0.75,0.25,0.5") == [0.75, 0.5, 0.25]


@pytest.mark.parametrize(
    ("ratio", "label"),
    [(1.0, "p100"), (0.75, "p075"), (0.5, "p050"), (0.25, "p025"), (0.0, "p000")],
)
def test_ratio_label(ratio: float, label: str) -> None:
    assert ratio_label(ratio) == label


def test_parse_ratios_rejects_out_of_range_values() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_ratios("1.1")
