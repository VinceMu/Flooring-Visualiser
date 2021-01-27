from enum import Enum, unique, auto


@unique
class QuiltMode(Enum):
    BEST = auto()
    CUT = auto(),
    RANDOM = auto()
