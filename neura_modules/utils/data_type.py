from enum import Enum

class DataType(str, Enum):
    BOOL = "bool"
    INT = "int"
    FLOAT = "float"
    STRING = "string"
    TUPLE_INT = "tuple_int"  # tuple[int]
    LIST_2D_FLOAT = "list_2d_float"  # list[list[float]]
    DICT_STRING_BOOL = "dict_string_bool"  # dict[str, bool]
    DICT_STRING_INT = "dict_string_int"  # dict[str, int]
    DICT_STRING_TUPLE_INT = "dict_string_tuple_int"  # dict[str, tuple[int]]


    def __str__(self) -> str:
        return str.__str__(self)
