# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import flutils.namedtupleutils as module_0
import collections as module_1


def test_case_0():
    none_type_0 = None
    module_0.to_namedtuple(none_type_0)


def test_case_1():
    str_0 = "The cherry-picking module attribute identifiers as the key. And the\n    value is the module name, which should be the key in ``modules``\n    "
    list_0 = [str_0, str_0, str_0]
    var_0 = module_0.to_namedtuple(list_0)
    bool_0 = True
    tuple_0 = (list_0, var_0, bool_0)
    var_1 = module_0.to_namedtuple(tuple_0)


def test_case_2():
    ordered_dict_0 = module_1.OrderedDict()
    var_0 = module_0.to_namedtuple(ordered_dict_0)


def test_case_3():
    str_0 = "The cherry-picking module attribute identifiers as the key. And the\n    value is the module name, which should be the key in ``modules``\n    "
    module_0.to_namedtuple(str_0)


def test_case_4():
    list_0 = []
    var_0 = module_0.to_namedtuple(list_0)


def test_case_5():
    str_0 = "Q3TJWxSPGd"
    dict_0 = {str_0: str_0}
    var_0 = module_0.to_namedtuple(dict_0)
    var_1 = module_0.to_namedtuple(var_0)


def test_case_6():
    list_0 = []
    ordered_dict_0 = module_1.OrderedDict(*list_0)
    var_0 = module_0.to_namedtuple(ordered_dict_0)
    var_1 = module_0.to_namedtuple(var_0)


def test_case_7():
    tuple_0 = ()
    var_0 = module_0.to_namedtuple(tuple_0)


def test_case_8():
    str_0 = "The cherry-picking module attribute identifiers as the key. And the\n    value is the module name, which should be the key in ``modules``\n    "
    list_0 = [str_0, str_0, str_0]
    var_0 = module_0.to_namedtuple(list_0)
    tuple_0 = (list_0, var_0, var_0)
    var_1 = module_0.to_namedtuple(tuple_0)


def test_case_9():
    none_type_0 = None
    bytes_0 = b"f"
    tuple_0 = (bytes_0,)
    dict_0 = {none_type_0: tuple_0, none_type_0: none_type_0, bytes_0: bytes_0}
    dict_1 = {none_type_0: none_type_0, none_type_0: dict_0}
    var_0 = module_0.to_namedtuple(dict_1)
    module_0.to_namedtuple(none_type_0)


def test_case_10():
    str_0 = "Q3TJW-x4PGd"
    dict_0 = {str_0: str_0}
    ordered_dict_0 = module_1.OrderedDict(**dict_0)
    var_0 = module_0.to_namedtuple(dict_0)


def test_case_11():
    none_type_0 = None
    bytes_0 = b"f"
    tuple_0 = (bytes_0,)
    dict_0 = {
        tuple_0: tuple_0,
        none_type_0: tuple_0,
        none_type_0: none_type_0,
        bytes_0: bytes_0,
    }
    dict_1 = module_1.OrderedDict(**dict_0)
    module_0.to_namedtuple(dict_1)


def test_case_12():
    bool_0 = True
    str_0 = "\x0bQI"
    dict_0 = {str_0: str_0}
    ordered_dict_0 = module_1.OrderedDict(**dict_0)
    list_0 = [bool_0, bool_0, bool_0, ordered_dict_0]
    var_0 = module_0.to_namedtuple(list_0)
    module_1.OrderedDict(*list_0)
