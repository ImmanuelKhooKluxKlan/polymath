from __future__ import annotations

import numpy as np
import pytest

from ml.training.apply_raw_support_swap_harm_veto import harm_veto_scores


def test_harm_veto_can_only_preserve_or_zero_base_scores() -> None:
    base = np.asarray([0.91, 0.82, 0.40, 0.99])
    harm = np.asarray([0.10, 0.45, 0.90, 0.449])

    result = harm_veto_scores(base, harm, 0.45)

    assert result.tolist() == pytest.approx([0.91, 0.0, 0.0, 0.99])
    assert np.all(result <= base)


def test_harm_veto_rejects_mismatched_score_vectors() -> None:
    with pytest.raises(ValueError, match="Score shape mismatch"):
        harm_veto_scores(np.asarray([0.8]), np.asarray([0.2, 0.3]), 0.45)
