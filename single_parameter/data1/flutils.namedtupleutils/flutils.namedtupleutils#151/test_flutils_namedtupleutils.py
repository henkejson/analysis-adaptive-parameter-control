# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import flutils.namedtupleutils as module_0
import collections as module_1


def test_case_0():
    none_type_0 = None
    module_0.to_namedtuple(none_type_0)


def test_case_1():
    str_0 = "onPO"
    dict_0 = {str_0: str_0}
    var_0 = module_0.to_namedtuple(dict_0)
    bool_0 = False
    list_0 = [bool_0, dict_0, str_0, str_0]
    var_1 = module_0.to_namedtuple(list_0)
    var_2 = module_0.to_namedtuple(var_1)
    var_3 = module_0.to_namedtuple(var_0)


def test_case_2():
    str_0 = "onPO"
    dict_0 = {str_0: str_0}
    var_0 = module_0.to_namedtuple(dict_0)


def test_case_3():
    dict_0 = {}
    var_0 = module_0.to_namedtuple(dict_0)


def test_case_4():
    str_0 = "9W.4N"
    module_0.to_namedtuple(str_0)


def test_case_5():
    tuple_0 = ()
    var_0 = module_0.to_namedtuple(tuple_0)


def test_case_6():
    ordered_dict_0 = module_1.OrderedDict()
    var_0 = module_0.to_namedtuple(ordered_dict_0)


def test_case_7():
    str_0 = "onPO"
    dict_0 = {str_0: str_0}
    var_0 = module_0.to_namedtuple(dict_0)
    var_1 = module_0.to_namedtuple(var_0)


def test_case_8():
    str_0 = "onPO,"
    dict_0 = {str_0: str_0, str_0: str_0}
    var_0 = module_0.to_namedtuple(dict_0)


def test_case_9():
    str_0 = ""
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0}
    tuple_0 = (str_0, dict_0, str_0)
    var_0 = module_0.to_namedtuple(tuple_0)
    var_1 = module_0.to_namedtuple(var_0)


def test_case_10():
    list_0 = []
    var_0 = module_0.to_namedtuple(list_0)


def test_case_11():
    str_0 = "V"
    dict_0 = {str_0: str_0, str_0: str_0}
    tuple_0 = (str_0, dict_0, str_0)
    var_0 = module_0.to_namedtuple(tuple_0)


def test_case_12():
    bool_0 = False
    int_0 = -1717
    dict_0 = {bool_0: bool_0, int_0: int_0}
    var_0 = module_0.to_namedtuple(dict_0)


def test_case_13():
    bytes_0 = b']s{\xaeX\x03\xe6"\xf0\x0e\x12\ns\xbeb\x9a`'
    tuple_0 = (bytes_0,)
    dict_0 = {bytes_0: bytes_0, tuple_0: tuple_0}
    int_0 = -1227
    tuple_1 = (tuple_0, dict_0, int_0)
    var_0 = module_0.to_namedtuple(tuple_0)
    module_0.to_namedtuple(tuple_1)


def test_case_14():
    str_0 = "onP\x0c"
    dict_0 = {str_0: str_0}
    tuple_0 = (str_0, dict_0)
    list_0 = [tuple_0, str_0, str_0, str_0]
    var_0 = module_0.to_namedtuple(list_0)
    var_1 = module_0.to_namedtuple(var_0)
    module_0.to_namedtuple(str_0)
