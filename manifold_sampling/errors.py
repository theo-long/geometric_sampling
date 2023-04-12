from enum import IntEnum

class ConstraintError(Exception):
    pass

class RejectCode(IntEnum):
    NONE = 0
    INEQUALITY = 1
    PROJECTION = 2
    REVERSE_PROJECTION = 3
    MH = 4
