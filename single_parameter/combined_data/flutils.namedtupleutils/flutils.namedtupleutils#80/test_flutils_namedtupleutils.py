# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import flutils.namedtupleutils as module_0
import collections as module_1


def test_case_0():
    none_type_0 = None
    module_0.to_namedtuple(none_type_0)


def test_case_1():
    ordered_dict_0 = module_1.OrderedDict()
    bytes_0 = b"\xc5\x92"
    str_0 = "cached_property"
    bool_0 = True
    tuple_0 = (ordered_dict_0, bytes_0, str_0, bool_0)
    var_0 = module_0.to_namedtuple(tuple_0)


def test_case_2():
    str_0 = "aHo"
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0}
    var_0 = module_0.to_namedtuple(dict_0)
    list_0 = [dict_0, dict_0, var_0, var_0]
    var_1 = module_0.to_namedtuple(list_0)


def test_case_3():
    dict_0 = {}
    var_0 = module_0.to_namedtuple(dict_0)


def test_case_4():
    str_0 = "f9L3K`Q>kRF6HYp"
    module_0.to_namedtuple(str_0)


def test_case_5():
    tuple_0 = ()
    var_0 = module_0.to_namedtuple(tuple_0)


def test_case_6():
    float_0 = -2495.2295
    dict_0 = {float_0: float_0}
    var_0 = module_0.to_namedtuple(dict_0)
    var_1 = module_1.OrderedDict(**dict_0)
    var_2 = module_0.to_namedtuple(var_1)


def test_case_7():
    ordered_dict_0 = module_1.OrderedDict()
    var_0 = module_0.to_namedtuple(ordered_dict_0)


def test_case_8():
    ordered_dict_0 = module_1.OrderedDict()
    var_0 = module_0.to_namedtuple(ordered_dict_0)
    var_1 = module_0.to_namedtuple(var_0)


def test_case_9():
    list_0 = []
    var_0 = module_0.to_namedtuple(list_0)


def test_case_10():
    int_0 = 272
    list_0 = [int_0, int_0, int_0, int_0]
    str_0 = "aH("
    dict_0 = {str_0: list_0, str_0: list_0, str_0: list_0}
    var_0 = module_0.to_namedtuple(dict_0)
    bool_0 = False
    module_0.to_namedtuple(bool_0)


def test_case_11():
    int_0 = -3517
    list_0 = [int_0, int_0, int_0, int_0]
    str_0 = "\x0bao"
    dict_0 = {str_0: list_0, str_0: list_0, str_0: list_0}
    ordered_dict_0 = module_1.OrderedDict(**dict_0)
    list_1 = [list_0, list_0, ordered_dict_0, str_0]
    tuple_0 = (list_1, list_0)
    var_0 = module_0.to_namedtuple(tuple_0)
    module_1.namedtuple(str_0, list_0)


def test_case_12():
    bytes_0 = b"\x97ag>\tS\x03\xda\x16\xfe'\xb5\x91"
    dict_0 = {bytes_0: bytes_0, bytes_0: bytes_0}
    bool_0 = True
    tuple_0 = (bytes_0, dict_0, bool_0, bool_0)
    module_0.to_namedtuple(tuple_0)
