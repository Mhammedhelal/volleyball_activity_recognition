"""Data subpackage exports.

Expose label lists and standard split names for easy imports:

	from src.data import PERSON_ACTIONS, GROUP_ACTIVITIES, SUBGROUP_ACTIVITIES
	from src.data import TRAIN_VIDEOS, VAL_VIDEOS, TEST_VIDEOS

"""

from .labels import PERSON_ACTIONS, GROUP_ACTIVITIES, SUBGROUP_ACTIVITIES
from .splits import train_indecies as TRAIN_VIDEOS, val_indecies as VAL_VIDEOS, test_indecies as TEST_VIDEOS

__all__ = [
	"PERSON_ACTIONS",
	"GROUP_ACTIVITIES",
	"SUBGROUP_ACTIVITIES",
	"TRAIN_VIDEOS",
	"VAL_VIDEOS",
	"TEST_VIDEOS",
]
