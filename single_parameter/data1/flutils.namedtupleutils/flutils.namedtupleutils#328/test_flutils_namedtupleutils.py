# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import flutils.namedtupleutils as module_0
import collections as module_1
import builtins as module_2


def test_case_0():
    bool_0 = False
    module_0.to_namedtuple(bool_0)


def test_case_1():
    float_0 = -631.0
    list_0 = [float_0]
    bool_0 = False
    tuple_0 = (list_0, float_0, bool_0)
    var_0 = module_0.to_namedtuple(tuple_0)


def test_case_2():
    bool_0 = False
    dict_0 = {bool_0: bool_0}
    var_0 = module_0.to_namedtuple(dict_0)


def test_case_3():
    str_0 = "\x0c7+CBEux|D\\5)"
    module_0.to_namedtuple(str_0)


def test_case_4():
    tuple_0 = ()
    var_0 = module_0.to_namedtuple(tuple_0)


def test_case_5():
    tuple_0 = ()
    bool_0 = True
    str_0 = "zfo"
    dict_0 = {str_0: bool_0, bool_0: tuple_0, tuple_0: str_0}
    var_0 = module_0.to_namedtuple(dict_0)
    var_1 = module_0.to_namedtuple(var_0)


def test_case_6():
    str_0 = "hH~Ax"
    list_0 = [str_0]
    var_0 = module_0.to_namedtuple(list_0)
    bool_0 = True
    module_1.namedtuple(bool_0, bool_0, defaults=bool_0)


def test_case_7():
    bytes_0 = b'\xfe\x81\xdd\x13*YH\xa9\x81\x9629\xa1"R\x11H\xfc'
    bool_0 = True
    list_0 = [bytes_0, bytes_0, bytes_0, bool_0]
    dict_0 = {bytes_0: bytes_0, bytes_0: bool_0, bool_0: bytes_0, bool_0: list_0}
    module_0.to_namedtuple(dict_0)


def test_case_8():
    list_0 = []
    var_0 = module_0.to_namedtuple(list_0)


def test_case_9():
    dict_0 = {}
    var_0 = module_0.to_namedtuple(dict_0)


def test_case_10():
    dict_0 = {}
    var_0 = module_0.to_namedtuple(dict_0)
    var_1 = module_0.to_namedtuple(var_0)


def test_case_11():
    ordered_dict_0 = module_1.OrderedDict()
    var_0 = module_0.to_namedtuple(ordered_dict_0)


def test_case_12():
    str_0 = "9ZngWx|'mgS3mI"
    str_1 = 'i.Khl-"KsU'
    str_2 = 'l\tjq+"q'
    list_0 = [str_2]
    dict_0 = {str_0: str_0, str_1: str_1, str_1: str_0, str_2: list_0}
    ordered_dict_0 = module_1.OrderedDict(**dict_0)
    var_0 = module_0.to_namedtuple(ordered_dict_0)


def test_case_13():
    tuple_0 = ()
    var_0 = module_0.to_namedtuple(tuple_0)
    var_1 = module_0.to_namedtuple(tuple_0)
    var_2 = module_2.object()
    var_3 = module_0.to_namedtuple(tuple_0)
    ordered_dict_0 = module_1.OrderedDict(*var_1)
    var_4 = module_0.to_namedtuple(ordered_dict_0)
    dict_0 = {tuple_0: tuple_0, tuple_0: tuple_0, tuple_0: tuple_0, tuple_0: tuple_0}
    var_5 = module_0.to_namedtuple(dict_0)
    var_6 = module_0.to_namedtuple(var_1)
    var_7 = module_0.to_namedtuple(dict_0)
    var_8 = module_0.to_namedtuple(var_7)
    object_0 = module_2.object()
    var_9 = module_0.to_namedtuple(dict_0)
    var_10 = module_0.to_namedtuple(var_6)
    var_11 = module_0.to_namedtuple(var_1)
    list_0 = []
    var_12 = module_0.to_namedtuple(list_0)
    var_13 = module_0.to_namedtuple(tuple_0)
    bool_0 = False
    int_0 = -880
    tuple_1 = (bool_0, int_0, list_0)
    var_14 = module_0.to_namedtuple(tuple_1)
    str_0 = "yf "
    bool_1 = False
    var_15 = module_0.to_namedtuple(var_13)
    dict_1 = {str_0: var_11, bool_1: var_2, var_6: var_7}
    var_16 = module_0.to_namedtuple(dict_1)
    tuple_2 = (var_16,)
    var_17 = module_0.to_namedtuple(tuple_2)
    var_18 = module_0.to_namedtuple(var_5)
    module_0.to_namedtuple(bool_1)
