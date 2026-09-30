import numpy as np
import pytest

pytest.importorskip("scipy")

from ml.training.augment_texture_audit_audio_chroma import local_evidence, normalized


def test_normalized_zero_vector_is_finite():
    result = normalized(np.zeros(12))
    assert np.all(np.isfinite(result))
    assert result.sum() == 0.0


def test_local_evidence_preserves_dominant_pitch_class():
    chroma = np.zeros((4, 12), dtype=float)
    chroma[:, 9] = 10.0
    chroma[:, 0] = 1.0
    features = {
        "times": np.asarray([0.0, 0.1, 0.2, 0.3]),
        "chroma": chroma,
        "attack": chroma,
        "bass": chroma,
        "treble": chroma,
    }
    rows = local_evidence(features, 0.15, 0.08)
    assert rows[9]["meanChroma"] > rows[0]["meanChroma"]
    assert sum(row["meanChroma"] for row in rows) == pytest.approx(1.0)
