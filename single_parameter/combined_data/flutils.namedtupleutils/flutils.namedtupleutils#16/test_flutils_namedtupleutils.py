# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import flutils.namedtupleutils as module_0
import collections as module_1
import builtins as module_2


def test_case_0():
    float_0 = -1000.2
    module_0.to_namedtuple(float_0)


def test_case_1():
    bool_0 = False
    list_0 = [bool_0, bool_0]
    var_0 = module_0.to_namedtuple(list_0)


def test_case_2():
    bytes_0 = b"P\xf7\xaf\xc0\x01\xad\xddX\x8c\xf4\xa7\xbe\xe0\x9f\xdd\x8cN"
    dict_0 = {bytes_0: bytes_0, bytes_0: bytes_0, bytes_0: bytes_0}
    module_0.to_namedtuple(dict_0)


def test_case_3():
    ordered_dict_0 = module_1.OrderedDict()
    var_0 = module_0.to_namedtuple(ordered_dict_0)


def test_case_4():
    str_0 = "run"
    module_0.to_namedtuple(str_0)


def test_case_5():
    list_0 = []
    var_0 = module_0.to_namedtuple(list_0)


def test_case_6():
    int_0 = 5
    dict_0 = {int_0: int_0}
    var_0 = module_0.to_namedtuple(dict_0)


def test_case_7():
    str_0 = "/\\`}lj|a!wvrO="
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0, str_0: str_0}
    var_0 = module_0.to_namedtuple(dict_0)
    var_1 = module_0.to_namedtuple(var_0)


def test_case_8():
    str_0 = "3q"
    tuple_0 = (str_0,)
    var_0 = module_0.to_namedtuple(tuple_0)
    module_0.to_namedtuple(str_0)


def test_case_9():
    ordered_dict_0 = module_1.OrderedDict()
    var_0 = module_0.to_namedtuple(ordered_dict_0)
    var_1 = module_0.to_namedtuple(var_0)


def test_case_10():
    bool_0 = False
    str_0 = "cP"
    dict_0 = {str_0: bool_0, str_0: bool_0}
    ordered_dict_0 = module_1.OrderedDict(**dict_0)
    object_0 = module_2.object()
    var_0 = module_0.to_namedtuple(ordered_dict_0)
    var_1 = module_0.to_namedtuple(var_0)


def test_case_11():
    str_0 = "\nm"
    dict_0 = {str_0: str_0, str_0: str_0}
    ordered_dict_0 = module_1.OrderedDict()
    int_0 = 128
    bytes_0 = b";\xc0\x08_K\xee-yJ"
    tuple_0 = (dict_0, ordered_dict_0, int_0, bytes_0)
    var_0 = module_0.to_namedtuple(tuple_0)
    str_1 = "Item %r of the given 'cmd' is not of type 'str'.  Got: %r"
    module_0.to_namedtuple(str_1)
