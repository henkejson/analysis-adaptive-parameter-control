# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import flutils.namedtupleutils as module_0
import collections as module_1


def test_case_0():
    int_0 = 520
    module_0.to_namedtuple(int_0)


def test_case_1():
    int_0 = -3979
    tuple_0 = (int_0, int_0)
    var_0 = module_0.to_namedtuple(tuple_0)
    none_type_0 = None
    module_0.to_namedtuple(none_type_0)


def test_case_2():
    str_0 = ".}e0c o^"
    dict_0 = {str_0: str_0, str_0: str_0}
    var_0 = module_0.to_namedtuple(dict_0)


def test_case_3():
    ordered_dict_0 = module_1.OrderedDict()
    var_0 = module_0.to_namedtuple(ordered_dict_0)


def test_case_4():
    str_0 = "\\u{:0>4}"
    module_0.to_namedtuple(str_0)


def test_case_5():
    tuple_0 = ()
    var_0 = module_0.to_namedtuple(tuple_0)


def test_case_6():
    list_0 = []
    str_0 = "commands"
    dict_0 = {str_0: str_0}
    ordered_dict_0 = module_1.OrderedDict(*list_0, **dict_0)
    list_1 = [ordered_dict_0]
    var_0 = module_0.to_namedtuple(list_1)
    var_1 = module_0.to_namedtuple(var_0)


def test_case_7():
    str_0 = ".}e0c o^"
    dict_0 = {str_0: str_0, str_0: str_0}
    var_0 = module_0.to_namedtuple(dict_0)
    var_1 = module_0.to_namedtuple(var_0)


def test_case_8():
    bytes_0 = b"\xb3\x94\x97\xf8\xaa@\x19\xe7\x9c\x85)G\xf8\x0c\xab\x12\xa9\xdb"
    tuple_0 = ()
    dict_0 = {bytes_0: bytes_0, bytes_0: bytes_0, tuple_0: bytes_0, bytes_0: tuple_0}
    module_0.to_namedtuple(dict_0)


def test_case_9():
    list_0 = []
    var_0 = module_0.to_namedtuple(list_0)
    bytes_0 = b"\xa3\xaeNPq\xeb\xfc\xc2\x07\x9c\xaa\xc7O\xfc\xf2\xfd\x8aDc"
    module_0.to_namedtuple(bytes_0)


def test_case_10():
    str_0 = ".}e0c o^"
    dict_0 = {str_0: str_0, str_0: str_0}
    tuple_0 = (dict_0, dict_0)
    var_0 = module_0.to_namedtuple(tuple_0)


def test_case_11():
    float_0 = 4130.0
    dict_0 = {float_0: float_0}
    var_0 = module_0.to_namedtuple(dict_0)
    bytes_0 = b"\x9a\x18\x8f'\x035\xa9)-\"%\\_"
    module_0.to_namedtuple(bytes_0)


def test_case_12():
    str_0 = " dOx"
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0, str_0: str_0}
    ordered_dict_0 = module_1.OrderedDict(**dict_0)
    list_0 = [ordered_dict_0, dict_0]
    var_0 = module_0.to_namedtuple(list_0)
    var_1 = module_0.to_namedtuple(var_0)
    module_1.namedtuple(list_0, str_0, rename=ordered_dict_0)
