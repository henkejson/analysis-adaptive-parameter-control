# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import flutils.namedtupleutils as module_0
import collections as module_1


def test_case_0():
    none_type_0 = None
    module_0.to_namedtuple(none_type_0)


def test_case_1():
    bool_0 = False
    str_0 = "h"
    dict_0 = {bool_0: bool_0, str_0: str_0, str_0: str_0}
    tuple_0 = (bool_0, bool_0, dict_0, bool_0)
    var_0 = module_0.to_namedtuple(tuple_0)
    var_1 = module_0.to_namedtuple(var_0)


def test_case_2():
    bool_0 = False
    dict_0 = {bool_0: bool_0, bool_0: bool_0, bool_0: bool_0, bool_0: bool_0}
    var_0 = module_0.to_namedtuple(dict_0)


def test_case_3():
    ordered_dict_0 = module_1.OrderedDict()
    var_0 = module_0.to_namedtuple(ordered_dict_0)


def test_case_4():
    bytes_0 = b"\xa8Ss\xa9\xaf.\x13.W\no\xc6"
    module_0.to_namedtuple(bytes_0)


def test_case_5():
    list_0 = []
    var_0 = module_0.to_namedtuple(list_0)


def test_case_6():
    dict_0 = {}
    var_0 = module_0.to_namedtuple(dict_0)


def test_case_7():
    bool_0 = False
    str_0 = "h"
    dict_0 = {bool_0: bool_0, str_0: str_0, str_0: str_0}
    var_0 = module_0.to_namedtuple(dict_0)


def test_case_8():
    tuple_0 = ()
    dict_0 = {tuple_0: tuple_0, tuple_0: tuple_0, tuple_0: tuple_0}
    var_0 = module_0.to_namedtuple(dict_0)
    var_1 = module_0.to_namedtuple(var_0)


def test_case_9():
    tuple_0 = ()
    var_0 = module_0.to_namedtuple(tuple_0)


def test_case_10():
    str_0 = "PT"
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0}
    ordered_dict_0 = module_1.OrderedDict(**dict_0)
    list_0 = [ordered_dict_0]
    var_0 = module_0.to_namedtuple(list_0)


def test_case_11():
    str_0 = '!61d"!F$O,nd'
    dict_0 = {str_0: str_0}
    ordered_dict_0 = module_1.OrderedDict(**dict_0)
    var_0 = module_0.to_namedtuple(ordered_dict_0)


def test_case_12():
    str_0 = "PT"
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0}
    var_0 = module_0.to_namedtuple(dict_0)
    var_1 = module_0.to_namedtuple(var_0)


def test_case_13():
    bool_0 = False
    bytes_0 = b"S\xf5\x9c\x10 \x87%\xa7"
    dict_0 = {bool_0: bool_0, bool_0: bytes_0, bytes_0: bytes_0}
    tuple_0 = (dict_0, bool_0)
    tuple_1 = (bool_0, bool_0, bytes_0, tuple_0)
    module_0.to_namedtuple(tuple_1)


def test_case_14():
    bool_0 = False
    str_0 = "\x0buJqa"
    dict_0 = {bool_0: bool_0, str_0: str_0, str_0: str_0}
    tuple_0 = (bool_0, bool_0, dict_0, bool_0)
    var_0 = module_0.to_namedtuple(tuple_0)
    ordered_dict_0 = module_1.OrderedDict()
    var_1 = module_0.to_namedtuple(var_0)
    set_0 = set()
    module_0.to_namedtuple(set_0)
