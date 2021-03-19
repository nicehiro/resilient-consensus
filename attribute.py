from enum import Enum


class Attribute(Enum):
    """Property of node.

    NORMAL node
    RANDOM value node
    CONSTANT value node
    """

    NORMAL = 1
    RANDOM = 2
    CONSTANT = 3
    INTELLIGENT = 4
