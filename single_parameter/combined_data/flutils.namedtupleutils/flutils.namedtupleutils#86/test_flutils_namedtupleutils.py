# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import flutils.namedtupleutils as module_0
import collections as module_1


def test_case_0():
    none_type_0 = None
    module_0.to_namedtuple(none_type_0)


def test_case_1():
    int_0 = -2444
    list_0 = [int_0]
    var_0 = module_0.to_namedtuple(list_0)
    var_1 = module_0.to_namedtuple(var_0)


def test_case_2():
    ordered_dict_0 = module_1.OrderedDict()
    var_0 = module_0.to_namedtuple(ordered_dict_0)


def test_case_3():
    str_0 = "is_list_like"
    module_0.to_namedtuple(str_0)


def test_case_4():
    tuple_0 = ()
    var_0 = module_0.to_namedtuple(tuple_0)


def test_case_5():
    dict_0 = {}
    var_0 = module_0.to_namedtuple(dict_0)
    var_1 = module_0.to_namedtuple(var_0)


def test_case_6():
    str_0 = "b"
    bool_0 = False
    dict_0 = {str_0: str_0, bool_0: str_0, bool_0: str_0, str_0: str_0}
    tuple_0 = (str_0, str_0, dict_0)
    var_0 = module_0.to_namedtuple(tuple_0)


def test_case_7():
    list_0 = []
    var_0 = module_0.to_namedtuple(list_0)


def test_case_8():
    str_0 = "b"
    bool_0 = False
    dict_0 = {str_0: str_0, bool_0: str_0, bool_0: str_0, str_0: str_0}
    tuple_0 = (str_0, str_0, dict_0)
    var_0 = module_0.to_namedtuple(tuple_0)
    var_1 = module_0.to_namedtuple(var_0)


def test_case_9():
    str_0 = "is_list_lFke"
    ordered_dict_0 = module_1.OrderedDict()
    tuple_0 = (ordered_dict_0, str_0, ordered_dict_0, ordered_dict_0)
    var_0 = module_0.to_namedtuple(tuple_0)
    str_1 = "_^aUrgQXW"
    dict_0 = {str_1: var_0, str_0: str_0, str_0: str_1}
    ordered_dict_1 = module_1.OrderedDict(**dict_0)
    var_1 = module_0.to_namedtuple(ordered_dict_1)


def test_case_10():
    bytes_0 = b"\x97\x90c\x99\x1c-R\x96"
    dict_0 = {bytes_0: bytes_0, bytes_0: bytes_0}
    module_0.to_namedtuple(dict_0)


def test_case_11():
    str_0 = "\x0bsulist_lFke"
    bool_0 = False
    dict_0 = {
        str_0: str_0,
        str_0: str_0,
        bool_0: bool_0,
        str_0: str_0,
        str_0: bool_0,
        bool_0: str_0,
    }
    var_0 = module_0.to_namedtuple(dict_0)
    ordered_dict_0 = module_1.OrderedDict()
    tuple_0 = (ordered_dict_0, str_0, ordered_dict_0, ordered_dict_0)
    tuple_1 = (str_0, tuple_0, ordered_dict_0)
    var_1 = module_0.to_namedtuple(tuple_1)
    str_1 = "_^aUrgQXW"
    var_2 = module_0.to_namedtuple(var_1)
    module_1.namedtuple(tuple_0, var_1, module=str_1)
