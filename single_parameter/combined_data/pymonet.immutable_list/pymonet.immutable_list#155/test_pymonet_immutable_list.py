# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.immutable_list as module_0
import typing as module_1


def test_case_0():
    immutable_list_0 = module_0.ImmutableList()
    bool_0 = immutable_list_0.__eq__(immutable_list_0)
    var_0 = immutable_list_0.to_list()


def test_case_1():
    none_type_0 = None
    none_type_1 = None
    immutable_list_0 = module_0.ImmutableList(tail=none_type_1)
    immutable_list_1 = immutable_list_0.unshift(none_type_0)
    immutable_list_2 = immutable_list_0.unshift(immutable_list_1)
    immutable_list_3 = immutable_list_1.append(none_type_0)
    bool_0 = immutable_list_1.__eq__(none_type_1)


def test_case_2():
    none_type_0 = None
    none_type_1 = None
    none_type_2 = None
    immutable_list_0 = module_0.ImmutableList(is_empty=none_type_2)
    immutable_list_1 = immutable_list_0.append(none_type_1)
    immutable_list_2 = immutable_list_1.unshift(none_type_0)


def test_case_3():
    none_type_0 = None
    bool_0 = False
    immutable_list_0 = module_0.ImmutableList(is_empty=bool_0)
    immutable_list_1 = immutable_list_0.__add__(immutable_list_0)
    str_0 = immutable_list_1.__str__()
    bool_1 = immutable_list_0.__eq__(bool_0)
    var_0 = immutable_list_0.reduce(none_type_0, immutable_list_0)
    immutable_list_0.__add__(none_type_0)


def test_case_4():
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(none_type_0, none_type_0)
    str_0 = immutable_list_0.__len__()
    immutable_list_1 = module_0.ImmutableList(immutable_list_0)
    immutable_list_1.find(immutable_list_0)


def test_case_5():
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(is_empty=none_type_0)
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    var_0 = immutable_list_0.to_list()
    var_1 = immutable_list_1.__len__()
    var_2 = var_0.__len__()
    immutable_list_1.find(var_1)


def test_case_6():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.to_list()


def test_case_7():
    bytes_0 = b"\xdd\xc5\xeb:\x88\x87\xd2\x87\xb7\x84\x05\x8e\x87\xeb"
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_0.map(bytes_0)


def test_case_8():
    immutable_list_0 = module_0.ImmutableList()
    bool_0 = immutable_list_0.__eq__(immutable_list_0)
    var_0 = immutable_list_0.find(immutable_list_0)
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    immutable_list_2 = module_0.ImmutableList(immutable_list_0)
    immutable_list_1.map(var_0)


def test_case_9():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_0.filter(immutable_list_0)


def test_case_10():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.find(immutable_list_0)
    immutable_list_0.filter(immutable_list_0)


def test_case_11():
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(none_type_0, none_type_0)
    immutable_list_1 = module_0.ImmutableList(immutable_list_0)
    immutable_list_1.find(immutable_list_0)


def test_case_12():
    bool_0 = True
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(tail=none_type_0, is_empty=none_type_0)
    var_0 = immutable_list_0.reduce(bool_0, bool_0)


def test_case_13():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    str_0 = immutable_list_0.__str__()
    var_0 = immutable_list_1.__len__()
    immutable_list_2 = immutable_list_0.__add__(immutable_list_1)
    immutable_list_1.reduce(immutable_list_0, immutable_list_2)


def test_case_14():
    immutable_list_0 = module_0.ImmutableList()


def test_case_15():
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(none_type_0, none_type_0)
    str_0 = immutable_list_0.__str__()
    immutable_list_1 = module_0.ImmutableList(immutable_list_0)
    immutable_list_1.find(immutable_list_0)


def test_case_16():
    bool_0 = True
    immutable_list_0 = module_0.ImmutableList(is_empty=bool_0)
    generic_0 = module_1.Generic()
    immutable_list_1 = module_0.ImmutableList(generic_0)
    var_0 = immutable_list_1.__len__()
    var_1 = immutable_list_0.reduce(var_0, immutable_list_0)
    var_2 = immutable_list_0.reduce(var_1, immutable_list_0)
    var_3 = immutable_list_0.find(var_2)
    var_4 = immutable_list_0.reduce(bool_0, immutable_list_0)
    immutable_list_0.filter(var_4)


def test_case_17():
    int_0 = -534
    none_type_0 = None
    tuple_0 = (int_0,)
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(tuple_0)
    immutable_list_1.find(none_type_0)


def test_case_18():
    str_0 = "j-t.-_N)u&v'4jB>+ZsO"
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(str_0)
    immutable_list_0.reduce(none_type_0, none_type_0)


def test_case_19():
    str_0 = "l\rhjhnwkI/ixKr"
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.append(str_0)
    immutable_list_2 = immutable_list_0.unshift(immutable_list_0)
    bool_0 = immutable_list_0.__eq__(immutable_list_1)
    immutable_list_2.find(str_0)


def test_case_20():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.__add__(immutable_list_0)
    var_0 = immutable_list_1.find(immutable_list_0)
    immutable_list_1.filter(immutable_list_1)


def test_case_21():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    bool_0 = immutable_list_1.__eq__(immutable_list_0)
    immutable_list_1.find(immutable_list_1)
