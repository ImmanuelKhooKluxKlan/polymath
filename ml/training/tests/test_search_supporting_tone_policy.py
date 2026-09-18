import sys
from pathlib import Path


TRAINING_DIR = Path(__file__).resolve().parents[1]
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from search_supporting_tone_policy import onset_count  # noqa: E402


def test_onset_count_uses_millisecond_gesture_identity():
    assert onset_count(
        [
            {"time": 1.0},
            {"time": 1.0004},
            {"time": 1.1001},
        ]
    ) == 2
