# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import flutils.namedtupleutils as module_0
import collections as module_1


def test_case_0():
    none_type_0 = None
    module_0.to_namedtuple(none_type_0)


def test_case_1():
    ordered_dict_0 = module_1.OrderedDict()
    bool_0 = True
    tuple_0 = (ordered_dict_0, ordered_dict_0, ordered_dict_0, bool_0)
    var_0 = module_0.to_namedtuple(tuple_0)


def test_case_2():
    str_0 = '%u"lTbnPbv*t88t7B\x0b}'
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0, str_0: str_0}
    var_0 = module_0.to_namedtuple(dict_0)


def test_case_3():
    ordered_dict_0 = module_1.OrderedDict()
    var_0 = module_0.to_namedtuple(ordered_dict_0)


def test_case_4():
    bytes_0 = b"\x9c!\xb2\xef\x96\x9c"
    module_0.to_namedtuple(bytes_0)


def test_case_5():
    list_0 = []
    var_0 = module_0.to_namedtuple(list_0)


def test_case_6():
    str_0 = "patch"
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0, str_0: str_0}
    var_0 = module_0.to_namedtuple(dict_0)


def test_case_7():
    tuple_0 = ()
    var_0 = module_0.to_namedtuple(tuple_0)


def test_case_8():
    str_0 = "patch"
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0, str_0: str_0}
    var_0 = module_0.to_namedtuple(dict_0)
    var_1 = module_0.to_namedtuple(var_0)
    var_2 = module_0.to_namedtuple(var_0)
    dict_1 = {var_2: dict_0, var_2: var_0, var_1: var_1}
    var_3 = module_0.to_namedtuple(dict_1)
    var_4 = module_0.to_namedtuple(dict_0)


def test_case_9():
    ordered_dict_0 = module_1.OrderedDict()
    var_0 = module_0.to_namedtuple(ordered_dict_0)
    var_1 = module_0.to_namedtuple(var_0)


def test_case_10():
    str_0 = "patch"
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0, str_0: str_0}
    var_0 = module_0.to_namedtuple(dict_0)
    var_1 = module_0.to_namedtuple(var_0)


def test_case_11():
    str_0 = "atc "
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0, str_0: str_0}
    var_0 = module_0.to_namedtuple(dict_0)
    var_1 = module_0.to_namedtuple(var_0)
    ordered_dict_0 = module_1.OrderedDict()
    var_2 = module_0.to_namedtuple(ordered_dict_0)
    module_1.namedtuple(var_0, var_2, module=dict_0)


def test_case_12():
    bytes_0 = b"\t\x1a\x96\xfc4\x06\xdc\x11\x1c\x02\xbe\xfa\x99a"
    dict_0 = {bytes_0: bytes_0, bytes_0: bytes_0, bytes_0: bytes_0}
    module_0.to_namedtuple(dict_0)
