# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import flutils.namedtupleutils as module_0
import collections as module_1


def test_case_0():
    none_type_0 = None
    module_0.to_namedtuple(none_type_0)


def test_case_1():
    set_0 = set()
    bytes_0 = b"\xb9\xf3"
    int_0 = -680
    list_0 = [set_0, bytes_0, int_0, bytes_0]
    tuple_0 = (set_0, bytes_0, set_0, list_0)
    tuple_1 = (tuple_0, set_0)
    var_0 = module_0.to_namedtuple(tuple_1)


def test_case_2():
    ordered_dict_0 = module_1.OrderedDict()
    var_0 = module_0.to_namedtuple(ordered_dict_0)


def test_case_3():
    str_0 = "create_module"
    module_0.to_namedtuple(str_0)


def test_case_4():
    tuple_0 = ()
    var_0 = module_0.to_namedtuple(tuple_0)


def test_case_5():
    dict_0 = {}
    var_0 = module_0.to_namedtuple(dict_0)


def test_case_6():
    str_0 = "lhxa"
    dict_0 = {str_0: str_0}
    var_0 = module_0.to_namedtuple(dict_0)
    var_1 = module_0.to_namedtuple(var_0)


def test_case_7():
    dict_0 = {}
    tuple_0 = module_0.to_namedtuple(dict_0)
    var_0 = module_0.to_namedtuple(tuple_0)


def test_case_8():
    list_0 = []
    var_0 = module_0.to_namedtuple(list_0)


def test_case_9():
    dict_0 = {}
    tuple_0 = (dict_0, dict_0)
    var_0 = module_0.to_namedtuple(tuple_0)


def test_case_10():
    bool_0 = False
    complex_0 = -949.7 + 639j
    dict_0 = {bool_0: bool_0, complex_0: complex_0}
    var_0 = module_0.to_namedtuple(dict_0)


def test_case_11():
    str_0 = "The cherry-picking module attribute identifiers as the key. And the\n    value is the module name, which should be the key in ``modules``\n    "
    dict_0 = {str_0: str_0, str_0: str_0}
    tuple_0 = ()
    tuple_1 = (str_0, str_0, dict_0, tuple_0)
    var_0 = module_0.to_namedtuple(tuple_1)
    str_1 = "create_module"
    module_0.to_namedtuple(str_1)


def test_case_12():
    bytes_0 = b"$?"
    tuple_0 = (bytes_0,)
    dict_0 = {bytes_0: bytes_0, bytes_0: bytes_0, bytes_0: tuple_0, tuple_0: tuple_0}
    tuple_1 = (bytes_0, dict_0)
    module_0.to_namedtuple(tuple_1)


def test_case_13():
    dict_0 = {}
    float_0 = 4266.0
    tuple_0 = (dict_0, float_0, dict_0, float_0)
    var_0 = module_0.to_namedtuple(tuple_0)
    str_0 = " hxa"
    dict_1 = {str_0: str_0}
    ordered_dict_0 = module_1.OrderedDict(**dict_1)
    var_1 = module_0.to_namedtuple(ordered_dict_0)
    var_2 = module_0.to_namedtuple(var_1)
    var_3 = module_0.to_namedtuple(var_0)
    module_0.to_namedtuple(str_0)
