# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import flutils.namedtupleutils as module_0
import collections as module_1
import builtins as module_2


def test_case_0():
    bool_0 = False
    module_0.to_namedtuple(bool_0)


def test_case_1():
    str_0 = "M"
    bool_0 = True
    dict_0 = {str_0: bool_0}
    ordered_dict_0 = module_1.OrderedDict(**dict_0)
    var_0 = module_0.to_namedtuple(dict_0)


def test_case_2():
    str_0 = "M"
    dict_0 = {str_0: str_0}
    var_0 = module_0.to_namedtuple(dict_0)
    var_1 = module_0.to_namedtuple(var_0)


def test_case_3():
    ordered_dict_0 = module_1.OrderedDict()
    var_0 = module_0.to_namedtuple(ordered_dict_0)


def test_case_4():
    bytes_0 = b"KL\x90Va\x15\xec\x96\xbb\xb7\xa5c\xa1\x0b\x0c\xdfSe\xee"
    module_0.to_namedtuple(bytes_0)


def test_case_5():
    list_0 = []
    var_0 = module_0.to_namedtuple(list_0)


def test_case_6():
    int_0 = 2
    dict_0 = {int_0: int_0, int_0: int_0, int_0: int_0, int_0: int_0}
    str_0 = "metadata"
    bytes_0 = b"_.\x81\xb0@\xe8"
    float_0 = -1102.759015
    tuple_0 = (bytes_0, float_0)
    tuple_1 = (dict_0, str_0, tuple_0, str_0)
    var_0 = module_0.to_namedtuple(tuple_1)
    tuple_2 = ()
    var_1 = module_0.to_namedtuple(tuple_2)
    var_2 = module_0.to_namedtuple(var_1)


def test_case_7():
    dict_0 = {}
    tuple_0 = module_0.to_namedtuple(dict_0)


def test_case_8():
    ordered_dict_0 = module_1.OrderedDict()
    var_0 = module_0.to_namedtuple(ordered_dict_0)
    var_1 = module_0.to_namedtuple(ordered_dict_0)
    var_2 = module_0.to_namedtuple(ordered_dict_0)
    var_3 = module_0.to_namedtuple(var_2)


def test_case_9():
    tuple_0 = ()
    var_0 = module_0.to_namedtuple(tuple_0)


def test_case_10():
    dict_0 = {}
    tuple_0 = ()
    tuple_1 = (dict_0, tuple_0, tuple_0, tuple_0)
    var_0 = module_0.to_namedtuple(tuple_1)


def test_case_11():
    int_0 = -16
    dict_0 = {
        int_0: int_0,
        int_0: int_0,
        int_0: int_0,
        int_0: int_0,
        int_0: int_0,
        int_0: int_0,
        int_0: int_0,
    }
    bytes_0 = b"_.\x81\xb0@\xe8"
    float_0 = -1102.759015
    tuple_0 = (bytes_0, float_0)
    dict_1 = {
        bytes_0: dict_0,
        int_0: tuple_0,
        int_0: tuple_0,
        int_0: bytes_0,
        int_0: tuple_0,
    }
    module_0.to_namedtuple(dict_1)


def test_case_12():
    object_0 = module_2.object()
    str_0 = "%!g"
    set_0 = {object_0, str_0}
    dict_0 = {object_0: str_0, object_0: set_0, str_0: set_0, str_0: set_0}
    ordered_dict_0 = module_1.OrderedDict(**dict_0)
    tuple_0 = ()
    var_0 = module_0.to_namedtuple(tuple_0)
    list_0 = [ordered_dict_0]
    var_1 = module_0.to_namedtuple(list_0)
    bool_0 = False
    dict_1 = {bool_0: bool_0, bool_0: bool_0, bool_0: bool_0}
    var_2 = module_0.to_namedtuple(dict_1)
    var_3 = module_0.to_namedtuple(dict_1)
    var_4 = module_0.to_namedtuple(var_3)
    object_1 = module_2.object()
    bool_1 = True
    list_1 = [var_3, dict_1, object_1, object_1]
    var_5 = module_0.to_namedtuple(list_1)
    var_6 = module_0.to_namedtuple(list_1)
    var_7 = module_0.to_namedtuple(var_3)
    var_8 = module_0.to_namedtuple(dict_1)
    var_9 = module_0.to_namedtuple(var_4)
    var_10 = module_0.to_namedtuple(var_3)
    var_11 = module_0.to_namedtuple(var_3)
    var_12 = module_0.to_namedtuple(var_4)
    module_0.to_namedtuple(bool_1)


def test_case_13():
    str_0 = "\tcM"
    set_0 = {str_0, str_0}
    dict_0 = {str_0: str_0, str_0: set_0, str_0: set_0, str_0: set_0}
    ordered_dict_0 = module_1.OrderedDict(**dict_0)
    list_0 = [ordered_dict_0]
    var_0 = module_0.to_namedtuple(list_0)
    bool_0 = False
    dict_1 = {bool_0: bool_0, bool_0: bool_0, bool_0: bool_0}
    var_1 = module_0.to_namedtuple(dict_1)
    var_2 = module_0.to_namedtuple(dict_1)
    ordered_dict_1 = module_1.OrderedDict()
    object_0 = module_2.object()
    list_1 = [list_0, var_2, dict_0, object_0, ordered_dict_1, var_0]
    var_3 = module_0.to_namedtuple(list_1)
    var_4 = module_0.to_namedtuple(ordered_dict_1)
    var_5 = module_0.to_namedtuple(var_2)
    module_0.to_namedtuple(str_0)
