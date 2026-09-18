import sys
from pathlib import Path


TRAINING_DIR = Path(__file__).resolve().parents[1]
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from search_selection_swap_policy import numeric_grid  # noqa: E402


def test_numeric_grid_is_copy_paste_friendly_and_deduplicated():
    assert numeric_grid("48, 52,48", integer=True) == [48, 52]
    assert numeric_grid("0.1, 0.25") == [0.1, 0.25]
