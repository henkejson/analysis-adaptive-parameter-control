# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import flutils.namedtupleutils as module_0
import builtins as module_1
import collections as module_2


def test_case_0():
    int_0 = 4
    module_0.to_namedtuple(int_0)


def test_case_1():
    bool_0 = False
    set_0 = {bool_0, bool_0, bool_0, bool_0}
    str_0 = "decode"
    object_0 = module_1.object()
    tuple_0 = (set_0, str_0, object_0)
    var_0 = module_0.to_namedtuple(tuple_0)


def test_case_2():
    str_0 = "decode"
    dict_0 = {str_0: str_0, str_0: str_0}
    var_0 = module_0.to_namedtuple(dict_0)


def test_case_3():
    ordered_dict_0 = module_2.OrderedDict()
    var_0 = module_0.to_namedtuple(ordered_dict_0)


def test_case_4():
    bytes_0 = b"\xd1\xd9"
    module_0.to_namedtuple(bytes_0)


def test_case_5():
    list_0 = []
    var_0 = module_0.to_namedtuple(list_0)


def test_case_6():
    object_0 = module_1.object()
    bytes_0 = b"\x1ao\xacL\x1b\x00"
    dict_0 = {object_0: object_0, bytes_0: bytes_0}
    module_0.to_namedtuple(dict_0)


def test_case_7():
    ordered_dict_0 = module_2.OrderedDict()
    list_0 = module_0.to_namedtuple(ordered_dict_0)
    var_0 = module_0.to_namedtuple(list_0)


def test_case_8():
    tuple_0 = ()
    var_0 = module_0.to_namedtuple(tuple_0)


def test_case_9():
    ordered_dict_0 = module_2.OrderedDict()
    list_0 = [ordered_dict_0, ordered_dict_0, ordered_dict_0, ordered_dict_0]
    var_0 = module_0.to_namedtuple(list_0)


def test_case_10():
    str_0 = "\n"
    str_1 = "1Q=0\n\n6eG+w\n$faRuU"
    dict_0 = {str_0: str_0, str_1: str_0}
    ordered_dict_0 = module_2.OrderedDict(**dict_0)
    int_0 = 20
    tuple_0 = (dict_0, ordered_dict_0, str_0)
    bool_0 = False
    tuple_1 = (tuple_0, int_0, tuple_0, bool_0)
    tuple_2 = (ordered_dict_0, int_0, int_0, tuple_1)
    tuple_3 = (tuple_2, bool_0, bool_0)
    var_0 = module_0.to_namedtuple(tuple_3)
    str_2 = "$m1ShxuB'h"
    module_0.to_namedtuple(str_2)


def test_case_11():
    str_0 = "decode"
    dict_0 = {str_0: str_0, str_0: str_0}
    var_0 = module_0.to_namedtuple(dict_0)
    var_1 = module_0.to_namedtuple(var_0)


def test_case_12():
    bool_0 = False
    set_0 = {bool_0, bool_0, bool_0, bool_0}
    str_0 = "decod "
    object_0 = module_1.object()
    dict_0 = {bool_0: bool_0, bool_0: bool_0, str_0: object_0}
    var_0 = module_0.to_namedtuple(dict_0)
    module_0.to_namedtuple(set_0)
