# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.immutable_list as module_0


def test_case_0():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    str_0 = immutable_list_1.__str__()
    bool_0 = immutable_list_0.__eq__(immutable_list_1)
    immutable_list_1.find(str_0)


def test_case_1():
    immutable_list_0 = module_0.ImmutableList()
    none_type_0 = None
    immutable_list_1 = immutable_list_0.unshift(none_type_0)


def test_case_2():
    bool_0 = True
    immutable_list_0 = module_0.ImmutableList(is_empty=bool_0)
    immutable_list_0.__add__(bool_0)


def test_case_3():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.__len__()


def test_case_4():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    str_0 = immutable_list_1.__len__()
    immutable_list_1.find(str_0)


def test_case_5():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.reduce(immutable_list_0, immutable_list_0)
    str_0 = immutable_list_0.__str__()


def test_case_6():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    str_0 = immutable_list_1.__str__()
    immutable_list_1.find(str_0)


def test_case_7():
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_0.map(none_type_0)


def test_case_8():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_0.filter(immutable_list_0)


def test_case_9():
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(none_type_0)
    immutable_list_2 = immutable_list_0.unshift(none_type_0)
    immutable_list_3 = immutable_list_0.append(none_type_0)
    var_0 = immutable_list_2.to_list()
    immutable_list_3.filter(var_0)


def test_case_10():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    str_0 = immutable_list_0.find(immutable_list_1)
    immutable_list_1.find(str_0)


def test_case_11():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    immutable_list_1.find(immutable_list_0)


def test_case_12():
    str_0 = "UWCgqIHU=Fdui%?"
    bool_0 = False
    immutable_list_0 = module_0.ImmutableList(str_0, is_empty=bool_0)
    immutable_list_1 = immutable_list_0.unshift(bool_0)
    immutable_list_0.reduce(str_0, str_0)


def test_case_13():
    immutable_list_0 = module_0.ImmutableList()


def test_case_14():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    immutable_list_2 = immutable_list_0.append(immutable_list_1)
    immutable_list_1.find(immutable_list_1)


def test_case_15():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    str_0 = immutable_list_1.__str__()
    bool_0 = immutable_list_0.__eq__(immutable_list_0)
    immutable_list_1.find(str_0)


def test_case_16():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    str_0 = immutable_list_1.__str__()
    immutable_list_2 = immutable_list_1.__add__(immutable_list_0)
    immutable_list_1.find(str_0)


def test_case_17():
    float_0 = 2661.1047358210562
    immutable_list_0 = module_0.ImmutableList(float_0)
    var_0 = immutable_list_0.__len__()
    immutable_list_0.find(var_0)


def test_case_18():
    float_0 = 2661.1047358210562
    immutable_list_0 = module_0.ImmutableList(float_0)
    immutable_list_0.find(immutable_list_0)


def test_case_19():
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(tail=none_type_0)
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    str_0 = immutable_list_1.__str__()
    immutable_list_2 = immutable_list_0.unshift(immutable_list_0)
    var_0 = immutable_list_1.to_list()
    immutable_list_2.map(immutable_list_0)


def test_case_20():
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(tail=none_type_0)
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    immutable_list_2 = immutable_list_1.to_list()
    str_0 = immutable_list_2.__str__()
    immutable_list_3 = immutable_list_0.unshift(immutable_list_2)
    immutable_list_4 = immutable_list_1.unshift(immutable_list_1)
    immutable_list_1.reduce(none_type_0, immutable_list_1)


def test_case_21():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    none_type_0 = None
    bool_0 = immutable_list_1.__eq__(none_type_0)
    var_0 = immutable_list_0.find(none_type_0)
    immutable_list_2 = immutable_list_1.unshift(var_0)
    str_0 = var_0.__str__()
    bool_1 = immutable_list_2.__eq__(immutable_list_0)
    immutable_list_1.find(immutable_list_0)
