# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.immutable_list as module_0


def test_case_0():
    immutable_list_0 = module_0.ImmutableList()
    bool_0 = immutable_list_0.__eq__(immutable_list_0)
    str_0 = immutable_list_0.__str__()
    var_0 = immutable_list_0.to_list()
    immutable_list_1 = var_0.__add__(var_0)
    var_1 = var_0.__len__()
    immutable_list_0.__add__(var_0)


def test_case_1():
    bool_0 = True
    immutable_list_0 = module_0.ImmutableList()
    bool_1 = immutable_list_0.__eq__(bool_0)
    immutable_list_1 = immutable_list_0.__add__(immutable_list_0)
    bool_2 = immutable_list_0.find(bool_1)
    str_0 = immutable_list_0.__str__()
    immutable_list_2 = immutable_list_0.unshift(immutable_list_0)
    immutable_list_2.reduce(immutable_list_2, bool_0)


def test_case_2():
    bool_0 = True
    immutable_list_0 = module_0.ImmutableList(bool_0, is_empty=bool_0)
    immutable_list_1 = immutable_list_0.append(bool_0)
    immutable_list_0.find(immutable_list_0)


def test_case_3():
    immutable_list_0 = module_0.ImmutableList()
    none_type_0 = None
    immutable_list_0.__add__(none_type_0)


def test_case_4():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.__len__()
    immutable_list_1 = module_0.ImmutableList(
        immutable_list_0, is_empty=immutable_list_0
    )
    immutable_list_1.find(immutable_list_1)


def test_case_5():
    bool_0 = False
    immutable_list_0 = module_0.ImmutableList(bool_0, is_empty=bool_0)
    str_0 = immutable_list_0.__len__()
    immutable_list_0.find(immutable_list_0)


def test_case_6():
    bool_0 = False
    immutable_list_0 = module_0.ImmutableList(bool_0, is_empty=bool_0)
    var_0 = immutable_list_0.to_list()
    immutable_list_0.find(var_0)


def test_case_7():
    immutable_list_0 = module_0.ImmutableList()
    none_type_0 = None
    var_0 = immutable_list_0.find(none_type_0)
    immutable_list_1 = immutable_list_0.append(immutable_list_0)
    var_1 = immutable_list_1.to_list()
    var_1.__add__(none_type_0)


def test_case_8():
    bool_0 = False
    immutable_list_0 = module_0.ImmutableList(is_empty=bool_0)
    var_0 = immutable_list_0.to_list()
    immutable_list_1 = module_0.ImmutableList()
    var_1 = immutable_list_0.to_list()
    immutable_list_0.map(var_1)


def test_case_9():
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(none_type_0, none_type_0)
    immutable_list_1 = immutable_list_0.unshift(none_type_0)
    immutable_list_2 = immutable_list_0.unshift(immutable_list_1)
    immutable_list_3 = immutable_list_1.append(none_type_0)
    immutable_list_3.map(none_type_0)


def test_case_10():
    bool_0 = False
    immutable_list_0 = module_0.ImmutableList(bool_0, is_empty=bool_0)
    immutable_list_0.filter(immutable_list_0)


def test_case_11():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.__add__(immutable_list_0)
    var_0 = immutable_list_1.__len__()
    immutable_list_1.filter(var_0)


def test_case_12():
    int_0 = 1947
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(none_type_0, none_type_0)
    var_0 = immutable_list_0.find(int_0)
    var_0.__len__()


def test_case_13():
    bool_0 = False
    immutable_list_0 = module_0.ImmutableList(bool_0, is_empty=bool_0)
    immutable_list_0.find(immutable_list_0)


def test_case_14():
    bool_0 = False
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.reduce(bool_0, immutable_list_0)
    immutable_list_1 = module_0.ImmutableList(bool_0, is_empty=bool_0)
    immutable_list_1.find(immutable_list_1)


def test_case_15():
    immutable_list_0 = module_0.ImmutableList()


def test_case_16():
    bool_0 = False
    immutable_list_0 = module_0.ImmutableList(bool_0, is_empty=bool_0)
    str_0 = immutable_list_0.__str__()
    immutable_list_0.find(immutable_list_0)


def test_case_17():
    bool_0 = False
    immutable_list_0 = module_0.ImmutableList(bool_0, is_empty=bool_0)
    var_0 = immutable_list_0.unshift(immutable_list_0)
    immutable_list_0.find(immutable_list_0)


def test_case_18():
    bool_0 = False
    immutable_list_0 = module_0.ImmutableList(is_empty=bool_0)
    immutable_list_1 = immutable_list_0.unshift(bool_0)
    immutable_list_2 = immutable_list_1.append(immutable_list_0)
    immutable_list_2.find(bool_0)


def test_case_19():
    bool_0 = True
    immutable_list_0 = module_0.ImmutableList(bool_0, is_empty=bool_0)
    immutable_list_1 = immutable_list_0.append(bool_0)
    var_0 = immutable_list_1.__len__()
    immutable_list_0.find(immutable_list_0)


def test_case_20():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.append(immutable_list_0)
    var_0 = immutable_list_0.reduce(immutable_list_1, immutable_list_1)
    bool_0 = immutable_list_1.__eq__(immutable_list_0)
    immutable_list_2 = immutable_list_1.unshift(immutable_list_1)
    str_0 = immutable_list_2.__str__()
    immutable_list_3 = module_0.ImmutableList(is_empty=str_0)
    immutable_list_4 = var_0.append(var_0)
    immutable_list_2.find(var_0)


def test_case_21():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    var_0 = immutable_list_0.reduce(immutable_list_1, immutable_list_1)
    var_1 = immutable_list_1.__len__()
    immutable_list_2 = immutable_list_1.unshift(immutable_list_1)
    bool_0 = immutable_list_2.__eq__(immutable_list_0)
    immutable_list_3 = module_0.ImmutableList(is_empty=var_1)
    var_0.find(var_0)


def test_case_22():
    bool_0 = True
    dict_0 = {}
    immutable_list_0 = module_0.ImmutableList(tail=dict_0)
    immutable_list_1 = module_0.ImmutableList(bool_0, is_empty=bool_0)
    var_0 = immutable_list_1.to_list()
    immutable_list_1.reduce(var_0, var_0)
