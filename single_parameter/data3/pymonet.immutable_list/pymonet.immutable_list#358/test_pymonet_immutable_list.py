# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.immutable_list as module_0
import typing as module_1


def test_case_0():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.__len__()
    bool_0 = immutable_list_0.__eq__(immutable_list_0)


def test_case_1():
    str_0 = "rAi8"
    immutable_list_0 = module_0.ImmutableList(str_0, is_empty=str_0)
    immutable_list_1 = immutable_list_0.append(str_0)
    bool_0 = immutable_list_1.__eq__(immutable_list_0)
    immutable_list_1.find(str_0)


def test_case_2():
    str_0 = "rAi8"
    immutable_list_0 = module_0.ImmutableList(str_0, is_empty=str_0)
    immutable_list_1 = immutable_list_0.__add__(immutable_list_0)
    immutable_list_0.find(immutable_list_0)


def test_case_3():
    bool_0 = False
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.to_list()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    immutable_list_2 = module_0.ImmutableList(bool_0, bool_0, bool_0)
    immutable_list_0.__add__(bool_0)


def test_case_4():
    str_0 = "rA8"
    immutable_list_0 = module_0.ImmutableList(str_0, is_empty=str_0)
    var_0 = immutable_list_0.__len__()
    immutable_list_0.find(var_0)


def test_case_5():
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(tail=none_type_0)
    var_0 = immutable_list_0.to_list()
    immutable_list_0.map(var_0)


def test_case_6():
    str_0 = "rAi8"
    immutable_list_0 = module_0.ImmutableList(str_0, is_empty=str_0)
    immutable_list_1 = immutable_list_0.append(str_0)
    str_1 = immutable_list_1.__str__()
    immutable_list_0.find(immutable_list_0)


def test_case_7():
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(none_type_0)
    immutable_list_2 = immutable_list_0.append(none_type_0)
    immutable_list_3 = immutable_list_2.unshift(none_type_0)
    immutable_list_2.map(immutable_list_2)


def test_case_8():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_0.filter(immutable_list_0)


def test_case_9():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.to_list()
    var_1 = immutable_list_0.reduce(var_0, var_0)
    immutable_list_1 = immutable_list_0.__add__(immutable_list_0)
    var_2 = immutable_list_0.find(var_0)
    immutable_list_1.filter(var_0)


def test_case_10():
    str_0 = "\n    First is a Monoid that will always return the first, value when 2 First instances are combined.\n    "
    immutable_list_0 = module_0.ImmutableList(tail=str_0)
    str_1 = immutable_list_0.find(str_0)


def test_case_11():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.reduce(immutable_list_0, immutable_list_0)
    immutable_list_0.filter(var_0)


def test_case_12():
    complex_0 = -5415.903181 - 3155.707j
    bool_0 = True
    bool_1 = False
    immutable_list_0 = module_0.ImmutableList(is_empty=bool_1)
    immutable_list_1 = immutable_list_0.unshift(bool_0)
    var_0 = immutable_list_0.find(complex_0)
    immutable_list_1.reduce(var_0, immutable_list_0)


def test_case_13():
    immutable_list_0 = module_0.ImmutableList()


def test_case_14():
    str_0 = "rAi8"
    immutable_list_0 = module_0.ImmutableList(str_0, is_empty=str_0)
    str_1 = immutable_list_0.__str__()
    immutable_list_0.find(immutable_list_0)


def test_case_15():
    str_0 = "rAi8"
    immutable_list_0 = module_0.ImmutableList(str_0, is_empty=str_0)
    immutable_list_1 = immutable_list_0.append(str_0)
    immutable_list_0.find(immutable_list_0)


def test_case_16():
    str_0 = "rAM8"
    immutable_list_0 = module_0.ImmutableList(str_0, is_empty=str_0)
    immutable_list_1 = immutable_list_0.unshift(str_0)
    immutable_list_0.find(immutable_list_0)


def test_case_17():
    str_0 = "rAi8"
    immutable_list_0 = module_0.ImmutableList(str_0, is_empty=str_0)
    immutable_list_1 = immutable_list_0.__add__(immutable_list_0)
    none_type_0 = None
    immutable_list_2 = immutable_list_1.__add__(immutable_list_1)
    immutable_list_0.find(none_type_0)


def test_case_18():
    str_0 = "rAi8"
    immutable_list_0 = module_0.ImmutableList(str_0, is_empty=str_0)
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    var_0 = immutable_list_1.__len__()
    immutable_list_1.find(var_0)


def test_case_19():
    int_0 = -2949
    generic_0 = module_1.Generic()
    dict_0 = {generic_0: generic_0, int_0: generic_0, generic_0: generic_0}
    immutable_list_0 = module_0.ImmutableList(tail=int_0, is_empty=dict_0)
    var_0 = immutable_list_0.find(generic_0)
    immutable_list_1 = module_0.ImmutableList(generic_0, is_empty=int_0)
    immutable_list_1.reduce(var_0, var_0)


def test_case_20():
    str_0 = "rAi8"
    immutable_list_0 = module_0.ImmutableList(str_0, is_empty=str_0)
    immutable_list_0.find(immutable_list_0)


def test_case_21():
    str_0 = "r<i8"
    immutable_list_0 = module_0.ImmutableList(str_0, is_empty=str_0)
    immutable_list_1 = immutable_list_0.unshift(str_0)
    immutable_list_1.find(str_0)


def test_case_22():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    bool_0 = immutable_list_1.__eq__(immutable_list_0)
    immutable_list_1.find(immutable_list_1)
