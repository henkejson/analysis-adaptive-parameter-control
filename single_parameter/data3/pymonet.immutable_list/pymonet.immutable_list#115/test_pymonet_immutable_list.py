# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.immutable_list as module_0


def test_case_0():
    bool_0 = False
    immutable_list_0 = module_0.ImmutableList(is_empty=bool_0)
    immutable_list_1 = immutable_list_0.unshift(bool_0)
    bool_1 = immutable_list_1.__eq__(immutable_list_1)
    var_0 = immutable_list_0.to_list()
    immutable_list_1.find(immutable_list_0)


def test_case_1():
    bool_0 = True
    immutable_list_0 = module_0.ImmutableList()
    bool_1 = immutable_list_0.__eq__(bool_0)
    immutable_list_1 = module_0.ImmutableList(bool_0, is_empty=bool_0)
    immutable_list_1.find(bool_0)


def test_case_2():
    bool_0 = True
    immutable_list_0 = module_0.ImmutableList(bool_0, is_empty=bool_0)
    immutable_list_1 = immutable_list_0.unshift(bool_0)
    immutable_list_0.find(bool_0)


def test_case_3():
    none_type_0 = None
    float_0 = -562.8
    immutable_list_0 = module_0.ImmutableList(is_empty=float_0)
    immutable_list_0.__add__(none_type_0)


def test_case_4():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.__len__()
    immutable_list_0.filter(var_0)


def test_case_5():
    bool_0 = False
    immutable_list_0 = module_0.ImmutableList(bool_0, is_empty=bool_0)
    str_0 = immutable_list_0.__len__()
    immutable_list_0.find(str_0)


def test_case_6():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.to_list()
    immutable_list_0.filter(var_0)


def test_case_7():
    bool_0 = True
    immutable_list_0 = module_0.ImmutableList(bool_0, is_empty=bool_0)
    immutable_list_1 = immutable_list_0.unshift(bool_0)
    str_0 = immutable_list_1.__str__()
    immutable_list_0.find(bool_0)


def test_case_8():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_0.map(immutable_list_0)


def test_case_9():
    immutable_list_0 = module_0.ImmutableList()
    none_type_0 = None
    immutable_list_1 = module_0.ImmutableList(none_type_0, none_type_0)
    immutable_list_2 = immutable_list_1.__add__(immutable_list_0)
    str_0 = immutable_list_2.__str__()
    list_0 = [immutable_list_2]
    immutable_list_2.map(list_0)


def test_case_10():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_0.filter(immutable_list_0)


def test_case_11():
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(is_empty=none_type_0)
    var_0 = immutable_list_0.find(none_type_0)
    immutable_list_0.filter(var_0)


def test_case_12():
    immutable_list_0 = module_0.ImmutableList()
    none_type_0 = None
    var_0 = immutable_list_0.reduce(immutable_list_0, none_type_0)
    var_1 = immutable_list_0.to_list()


def test_case_13():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.__add__(immutable_list_0)
    immutable_list_2 = immutable_list_1.append(immutable_list_1)
    immutable_list_3 = immutable_list_1.__add__(immutable_list_1)
    str_0 = immutable_list_0.__str__()
    immutable_list_4 = immutable_list_0.__add__(immutable_list_0)
    var_0 = immutable_list_0.reduce(immutable_list_0, immutable_list_0)
    immutable_list_5 = immutable_list_4.unshift(immutable_list_4)
    bool_0 = immutable_list_0.__eq__(immutable_list_0)
    immutable_list_6 = immutable_list_4.append(str_0)
    immutable_list_5.reduce(immutable_list_5, immutable_list_0)


def test_case_14():
    immutable_list_0 = module_0.ImmutableList()


def test_case_15():
    bool_0 = True
    immutable_list_0 = module_0.ImmutableList(bool_0, is_empty=bool_0)
    str_0 = immutable_list_0.__str__()
    immutable_list_0.find(str_0)


def test_case_16():
    bool_0 = False
    immutable_list_0 = module_0.ImmutableList(bool_0, is_empty=bool_0)
    immutable_list_1 = immutable_list_0.append(immutable_list_0)
    immutable_list_0.find(bool_0)


def test_case_17():
    bool_0 = True
    immutable_list_0 = module_0.ImmutableList(bool_0, is_empty=bool_0)
    immutable_list_1 = immutable_list_0.unshift(bool_0)
    immutable_list_2 = immutable_list_1.__add__(immutable_list_1)
    var_0 = immutable_list_0.to_list()
    immutable_list_0.find(var_0)


def test_case_18():
    bool_0 = True
    immutable_list_0 = module_0.ImmutableList(bool_0, is_empty=bool_0)
    bool_1 = immutable_list_0.__eq__(bool_0)
    bool_2 = immutable_list_0.__eq__(bool_0)
    immutable_list_0.reduce(bool_0, bool_0)


def test_case_19():
    bool_0 = False
    immutable_list_0 = module_0.ImmutableList(is_empty=bool_0)
    immutable_list_1 = immutable_list_0.unshift(bool_0)
    var_0 = immutable_list_0.find(bool_0)
    bool_1 = immutable_list_0.__eq__(immutable_list_0)
    var_1 = immutable_list_0.reduce(var_0, bool_1)
    immutable_list_1.filter(var_1)


def test_case_20():
    bool_0 = True
    immutable_list_0 = module_0.ImmutableList(bool_0, is_empty=bool_0)
    immutable_list_0.find(bool_0)


def test_case_21():
    bool_0 = True
    immutable_list_0 = module_0.ImmutableList(is_empty=bool_0)
    immutable_list_1 = immutable_list_0.unshift(bool_0)
    immutable_list_2 = immutable_list_1.__add__(immutable_list_0)
    var_0 = immutable_list_2.__len__()
    immutable_list_2.find(immutable_list_0)


def test_case_22():
    bool_0 = True
    immutable_list_0 = module_0.ImmutableList(bool_0, is_empty=bool_0)
    immutable_list_1 = immutable_list_0.append(bool_0)
    immutable_list_1.find(immutable_list_1)


def test_case_23():
    bool_0 = False
    immutable_list_0 = module_0.ImmutableList(is_empty=bool_0)
    immutable_list_1 = module_0.ImmutableList(is_empty=bool_0)
    immutable_list_2 = immutable_list_0.unshift(bool_0)
    var_0 = immutable_list_1.find(bool_0)
    bool_1 = immutable_list_1.__eq__(immutable_list_2)
    immutable_list_1.filter(var_0)


def test_case_24():
    bool_0 = False
    immutable_list_0 = module_0.ImmutableList(is_empty=bool_0)
    immutable_list_1 = immutable_list_0.unshift(bool_0)
    immutable_list_2 = immutable_list_1.__add__(immutable_list_0)
    bool_1 = immutable_list_1.__eq__(immutable_list_2)
    immutable_list_2.find(immutable_list_0)
