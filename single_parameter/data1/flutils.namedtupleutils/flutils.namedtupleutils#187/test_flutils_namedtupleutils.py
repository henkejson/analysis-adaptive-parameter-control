# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import flutils.namedtupleutils as module_0
import builtins as module_1
import collections as module_2


def test_case_0():
    none_type_0 = None
    module_0.to_namedtuple(none_type_0)


def test_case_1():
    list_0 = []
    object_0 = module_1.object(*list_0)
    list_1 = [object_0, list_0, list_0, object_0]
    var_0 = module_0.to_namedtuple(list_1)
    str_0 = "p]yP&\x0bT1"
    dict_0 = {str_0: str_0}
    var_1 = module_0.to_namedtuple(dict_0)


def test_case_2():
    ordered_dict_0 = module_2.OrderedDict()
    var_0 = module_0.to_namedtuple(ordered_dict_0)


def test_case_3():
    str_0 = "encoding"
    module_0.to_namedtuple(str_0)


def test_case_4():
    list_0 = []
    var_0 = module_0.to_namedtuple(list_0)


def test_case_5():
    dict_0 = {}
    var_0 = module_0.to_namedtuple(dict_0)


def test_case_6():
    str_0 = "subsequent_indent_len"
    dict_0 = {str_0: str_0}
    var_0 = module_0.to_namedtuple(dict_0)


def test_case_7():
    ordered_dict_0 = module_2.OrderedDict()
    var_0 = module_0.to_namedtuple(ordered_dict_0)
    var_1 = module_0.to_namedtuple(var_0)


def test_case_8():
    tuple_0 = ()
    var_0 = module_0.to_namedtuple(tuple_0)


def test_case_9():
    dict_0 = {}
    var_0 = module_0.to_namedtuple(dict_0)
    list_0 = [var_0, dict_0]
    var_1 = module_0.to_namedtuple(var_0)
    tuple_0 = (list_0,)
    var_2 = module_0.to_namedtuple(tuple_0)
    var_3 = module_0.to_namedtuple(tuple_0)


def test_case_10():
    str_0 = "p]yP&\x0bT1"
    dict_0 = {str_0: str_0}
    var_0 = module_0.to_namedtuple(dict_0)


def test_case_11():
    list_0 = []
    object_0 = module_1.object(*list_0)
    list_1 = [object_0, list_0, list_0, object_0]
    var_0 = module_0.to_namedtuple(list_1)
    str_0 = "p]yP&\x0bT1"
    dict_0 = {str_0: list_0, object_0: var_0, str_0: str_0}
    var_1 = module_0.to_namedtuple(dict_0)


def test_case_12():
    bytes_0 = b"?\xeb_"
    dict_0 = {bytes_0: bytes_0}
    tuple_0 = (dict_0,)
    module_0.to_namedtuple(tuple_0)


def test_case_13():
    str_0 = "subsequent_indent_len"
    dict_0 = {str_0: str_0}
    var_0 = module_0.to_namedtuple(dict_0)
    var_1 = module_0.to_namedtuple(var_0)


def test_case_14():
    bool_0 = False
    str_0 = "\x0ca"
    str_1 = "LWZB\\aQ<b5r[^"
    dict_0 = {str_0: str_0, str_0: str_0, str_1: bool_0}
    ordered_dict_0 = module_2.OrderedDict(**dict_0)
    var_0 = module_0.to_namedtuple(ordered_dict_0)
    var_1 = module_0.to_namedtuple(dict_0)
    tuple_0 = (bool_0,)
    var_2 = module_0.to_namedtuple(tuple_0)
    dict_1 = {tuple_0: tuple_0, bool_0: tuple_0, bool_0: tuple_0, tuple_0: tuple_0}
    var_3 = module_0.to_namedtuple(dict_1)
    var_4 = module_0.to_namedtuple(dict_0)
    var_5 = module_0.to_namedtuple(var_2)
    var_6 = module_0.to_namedtuple(var_0)
    bool_1 = True
    module_0.to_namedtuple(bool_1)
