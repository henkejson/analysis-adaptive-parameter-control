# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import flutils.namedtupleutils as module_0
import builtins as module_1
import collections as module_2


def test_case_0():
    bool_0 = True
    module_0.to_namedtuple(bool_0)


def test_case_1():
    object_0 = module_1.object()
    ordered_dict_0 = module_2.OrderedDict()
    var_0 = module_0.to_namedtuple(ordered_dict_0)
    list_0 = [var_0, object_0]
    var_1 = module_0.to_namedtuple(list_0)
    module_0.to_namedtuple(object_0)


def test_case_2():
    str_0 = "exec_module"
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0}
    ordered_dict_0 = module_2.OrderedDict(**dict_0)
    var_0 = module_0.to_namedtuple(ordered_dict_0)
    var_1 = module_0.to_namedtuple(var_0)


def test_case_3():
    ordered_dict_0 = module_2.OrderedDict()
    var_0 = module_0.to_namedtuple(ordered_dict_0)


def test_case_4():
    str_0 = "l_\x0c(;F^{saR `"
    module_0.to_namedtuple(str_0)


def test_case_5():
    tuple_0 = ()
    var_0 = module_0.to_namedtuple(tuple_0)


def test_case_6():
    bool_0 = True
    str_0 = "exec_module"
    dict_0 = {str_0: str_0, bool_0: bool_0, str_0: bool_0}
    ordered_dict_0 = module_2.OrderedDict(**dict_0)
    var_0 = module_0.to_namedtuple(ordered_dict_0)
    var_1 = module_0.to_namedtuple(var_0)


def test_case_7():
    dict_0 = {}
    var_0 = module_0.to_namedtuple(dict_0)


def test_case_8():
    ordered_dict_0 = module_2.OrderedDict()
    var_0 = module_0.to_namedtuple(ordered_dict_0)
    var_1 = module_0.to_namedtuple(var_0)


def test_case_9():
    bytes_0 = b"\x90W\xff\xdb\xd5Z\xee\x05\xd6\x01h\xbbz"
    int_0 = 448
    dict_0 = {bytes_0: bytes_0, bytes_0: bytes_0, int_0: bytes_0}
    tuple_0 = (bytes_0, dict_0, bytes_0, dict_0)
    module_0.to_namedtuple(tuple_0)


def test_case_10():
    list_0 = []
    var_0 = module_0.to_namedtuple(list_0)


def test_case_11():
    object_0 = module_1.object()
    str_0 = '=7,\\=e-aS\\&J"c'
    list_0 = [str_0, str_0, str_0]
    var_0 = module_0.to_namedtuple(list_0)


def test_case_12():
    str_0 = ""
    str_1 = "path"
    ordered_dict_0 = module_2.OrderedDict()
    dict_0 = {str_0: str_0, str_1: ordered_dict_0}
    ordered_dict_1 = module_0.to_namedtuple(dict_0)
    module_0.to_namedtuple(str_1)


def test_case_13():
    str_0 = "\x0chJ7"
    none_type_0 = None
    dict_0 = {str_0: str_0}
    var_0 = module_0.to_namedtuple(dict_0)
    module_2.namedtuple(str_0, none_type_0)
