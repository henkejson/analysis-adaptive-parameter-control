# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import flutils.namedtupleutils as module_0
import collections as module_1
import builtins as module_2


def test_case_0():
    int_0 = -2641
    module_0.to_namedtuple(int_0)


def test_case_1():
    int_0 = -441
    list_0 = [int_0, int_0]
    var_0 = module_0.to_namedtuple(list_0)
    var_1 = module_0.to_namedtuple(var_0)


def test_case_2():
    bool_0 = False
    dict_0 = {bool_0: bool_0, bool_0: bool_0}
    list_0 = [dict_0, dict_0, bool_0]
    var_0 = module_0.to_namedtuple(list_0)


def test_case_3():
    ordered_dict_0 = module_1.OrderedDict()
    var_0 = module_0.to_namedtuple(ordered_dict_0)


def test_case_4():
    bytes_0 = b"\x93\x9f>\xb1F\xa6cB\x1f\x0b\xc8"
    module_0.to_namedtuple(bytes_0)


def test_case_5():
    tuple_0 = ()
    var_0 = module_0.to_namedtuple(tuple_0)


def test_case_6():
    dict_0 = {}
    var_0 = module_0.to_namedtuple(dict_0)


def test_case_7():
    str_0 = "z"
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0, str_0: str_0}
    var_0 = module_0.to_namedtuple(dict_0)
    var_1 = module_0.to_namedtuple(var_0)


def test_case_8():
    ordered_dict_0 = module_1.OrderedDict()
    str_0 = "z"
    float_0 = 2733.209593798848
    dict_0 = {str_0: float_0, str_0: ordered_dict_0, float_0: ordered_dict_0}
    var_0 = module_0.to_namedtuple(dict_0)
    var_1 = module_0.to_namedtuple(var_0)


def test_case_9():
    list_0 = []
    var_0 = module_0.to_namedtuple(list_0)


def test_case_10():
    str_0 = "~Zpjwq,\\K"
    set_0 = {str_0, str_0}
    dict_0 = {str_0: str_0, str_0: str_0, str_0: set_0, str_0: set_0}
    var_0 = module_0.to_namedtuple(dict_0)


def test_case_11():
    str_0 = "U; i++?@IJ:_b6"
    tuple_0 = ()
    bool_0 = True
    var_0 = module_0.to_namedtuple(tuple_0)
    tuple_1 = (str_0, tuple_0, bool_0)
    dict_0 = {tuple_1: str_0, str_0: tuple_0, bool_0: bool_0, tuple_1: tuple_1}
    list_0 = [dict_0, dict_0, bool_0, tuple_1]
    ordered_dict_0 = module_1.OrderedDict()
    var_1 = module_0.to_namedtuple(ordered_dict_0)
    var_2 = module_0.to_namedtuple(list_0)
    var_3 = module_0.to_namedtuple(var_2)
    str_1 = " z"
    float_0 = 2733.3
    tuple_2 = (float_0, str_1)
    dict_1 = {}
    object_0 = module_2.object()
    str_2 = "utf8"
    dict_2 = {float_0: str_1, str_1: tuple_2, str_1: dict_1, object_0: str_2}
    var_4 = module_0.to_namedtuple(var_2)
    tuple_3 = (str_1, var_3, dict_2, dict_2)
    tuple_4 = (tuple_3,)
    var_5 = module_0.to_namedtuple(tuple_4)
    bytes_0 = b"&_\xdf\x18\xaeb\xd1\x1c\x82\x1f\xbd\x94\xfb"
    bool_1 = True
    set_0 = {bytes_0, bool_1}
    module_0.to_namedtuple(set_0)


def test_case_12():
    bytes_0 = b"\\\xa6v;\xae\x12\xff\xa3\x17X\x1e\x9d, \xb1\xb9.\xac"
    bytes_1 = b"\x8d\xf9j\x1f\x1bB26\xe4\x1e\xb8\x0c\xe5"
    float_0 = -1617.21088
    dict_0 = {bytes_0: bytes_0, bytes_1: bytes_1, bytes_0: float_0}
    module_0.to_namedtuple(dict_0)
