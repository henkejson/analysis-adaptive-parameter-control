# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import flutils.namedtupleutils as module_0
import collections as module_1


def test_case_0():
    none_type_0 = None
    module_0.to_namedtuple(none_type_0)


def test_case_1():
    bool_0 = True
    list_0 = [bool_0, bool_0, bool_0, bool_0]
    var_0 = module_0.to_namedtuple(list_0)


def test_case_2():
    str_0 = "is_loaded"
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0, str_0: str_0}
    var_0 = module_0.to_namedtuple(dict_0)
    var_1 = module_0.to_namedtuple(var_0)


def test_case_3():
    dict_0 = {}
    var_0 = module_0.to_namedtuple(dict_0)


def test_case_4():
    bytes_0 = b"\xbc\x80~\xd4\xf5\x94\xa1\x8d\xddH"
    module_0.to_namedtuple(bytes_0)


def test_case_5():
    list_0 = []
    var_0 = module_0.to_namedtuple(list_0)


def test_case_6():
    bool_0 = True
    dict_0 = {bool_0: bool_0, bool_0: bool_0}
    tuple_0 = (dict_0,)
    var_0 = module_0.to_namedtuple(tuple_0)
    bool_1 = False
    ordered_dict_0 = module_1.OrderedDict()
    module_0.to_namedtuple(bool_1)


def test_case_7():
    ordered_dict_0 = module_1.OrderedDict()
    var_0 = module_0.to_namedtuple(ordered_dict_0)


def test_case_8():
    bytes_0 = b"\xcf\x17\xe4\x1a\x87\xa1\x15m\xa1"
    bool_0 = False
    tuple_0 = (bytes_0, bytes_0, bool_0, bytes_0)
    var_0 = module_0.to_namedtuple(tuple_0)


def test_case_9():
    ordered_dict_0 = module_1.OrderedDict()
    list_0 = [ordered_dict_0, ordered_dict_0, ordered_dict_0]
    var_0 = module_0.to_namedtuple(list_0)
    var_1 = module_0.to_namedtuple(var_0)


def test_case_10():
    tuple_0 = ()
    var_0 = module_0.to_namedtuple(tuple_0)


def test_case_11():
    bytes_0 = b"\x99\xf8\t"
    str_0 = " !+V+Xcip"
    dict_0 = {str_0: bytes_0}
    ordered_dict_0 = module_1.OrderedDict(**dict_0)
    var_0 = module_0.to_namedtuple(dict_0)
    module_0.to_namedtuple(str_0)


def test_case_12():
    bytes_0 = b"\x99\xf8\t"
    float_0 = -488.5
    bool_0 = False
    str_0 = "y>]I}I]A=f1ha"
    dict_0 = {bytes_0: bytes_0}
    ordered_dict_0 = module_1.OrderedDict(**dict_0)
    tuple_0 = (str_0, ordered_dict_0, ordered_dict_0, ordered_dict_0)
    tuple_1 = (bytes_0, float_0, bool_0, tuple_0)
    module_0.to_namedtuple(tuple_1)


def test_case_13():
    float_0 = -2285.0
    list_0 = [float_0, float_0, float_0]
    var_0 = module_0.to_namedtuple(list_0)
    str_0 = "n4Vt\r"
    none_type_0 = None
    str_1 = "is_lodd"
    dict_0 = {
        str_0: none_type_0,
        str_1: none_type_0,
        none_type_0: none_type_0,
        str_1: none_type_0,
    }
    ordered_dict_0 = module_1.OrderedDict(**dict_0)
    var_1 = module_0.to_namedtuple(ordered_dict_0)
    var_2 = module_0.to_namedtuple(dict_0)
    var_3 = module_0.to_namedtuple(var_1)
    module_0.to_namedtuple(str_0)
