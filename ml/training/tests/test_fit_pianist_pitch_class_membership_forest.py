from __future__ import annotations

import numpy as np

from ml.training.pianist_pitch_class_decisions import membership_decision_mask


def test_membership_mask_includes_existing_or_source_supported_classes() -> None:
    arrays = {
        "candidate": np.asarray([[1, 0, 0, 1]], dtype=np.int8),
        "source": np.asarray([[0, 1, 0, 1]], dtype=np.int8),
    }

    assert membership_decision_mask(arrays).tolist() == [True, True, False, True]
