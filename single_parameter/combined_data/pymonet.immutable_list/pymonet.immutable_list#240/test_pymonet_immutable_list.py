# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.immutable_list as module_0


def test_case_0():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = module_0.ImmutableList(immutable_list_0)
    bool_0 = immutable_list_0.__eq__(immutable_list_1)
    immutable_list_1.find(immutable_list_0)


def test_case_1():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.find(immutable_list_0)
    bool_0 = immutable_list_0.__eq__(var_0)
    immutable_list_0.filter(var_0)


def test_case_2():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    immutable_list_1.find(immutable_list_1)


def test_case_3():
    none_type_0 = None
    complex_0 = 2495.971 + 2265.89319j
    immutable_list_0 = module_0.ImmutableList(tail=none_type_0)
    var_0 = immutable_list_0.to_list()
    var_1 = immutable_list_0.reduce(var_0, complex_0)
    var_2 = immutable_list_0.find(var_1)
    immutable_list_0.__add__(var_0)


def test_case_4():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.__len__()


def test_case_5():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    var_0 = immutable_list_1.__len__()
    immutable_list_1.find(var_0)


def test_case_6():
    bool_0 = False
    immutable_list_0 = module_0.ImmutableList(is_empty=bool_0)
    var_0 = immutable_list_0.to_list()
    var_1 = var_0.__len__()


def test_case_7():
    int_0 = -2628
    dict_0 = {}
    immutable_list_0 = module_0.ImmutableList(int_0, dict_0)
    immutable_list_0.to_list()


def test_case_8():
    int_0 = -2188
    immutable_list_0 = module_0.ImmutableList(is_empty=int_0)
    bool_0 = immutable_list_0.__eq__(int_0)
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    immutable_list_0.map(int_0)


def test_case_9():
    int_0 = 1
    set_0 = {int_0, int_0, int_0}
    bool_0 = False
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.reduce(bool_0, int_0)
    immutable_list_1 = module_0.ImmutableList()
    immutable_list_2 = immutable_list_1.unshift(bool_0)
    immutable_list_2.map(set_0)


def test_case_10():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_0.filter(immutable_list_0)


def test_case_11():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.find(immutable_list_0)
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    immutable_list_1.filter(immutable_list_1)


def test_case_12():
    str_0 = "@mWK)ZwHSOY<"
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.find(str_0)


def test_case_13():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.find(immutable_list_0)
    bool_0 = immutable_list_0.__eq__(var_0)
    var_1 = immutable_list_0.reduce(immutable_list_0, var_0)
    immutable_list_0.filter(var_0)


def test_case_14():
    bool_0 = True
    immutable_list_0 = module_0.ImmutableList(bool_0)
    immutable_list_1 = immutable_list_0.append(immutable_list_0)
    none_type_0 = None
    bool_1 = False
    immutable_list_2 = immutable_list_0.append(bool_1)
    immutable_list_0.reduce(bool_0, none_type_0)


def test_case_15():
    immutable_list_0 = module_0.ImmutableList()


def test_case_16():
    immutable_list_0 = module_0.ImmutableList()
    str_0 = immutable_list_0.__str__()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    immutable_list_1.find(immutable_list_1)


def test_case_17():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.find(immutable_list_0)
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    immutable_list_2 = immutable_list_1.append(immutable_list_0)
    immutable_list_1.filter(immutable_list_1)


def test_case_18():
    immutable_list_0 = module_0.ImmutableList()
    bool_0 = immutable_list_0.__eq__(immutable_list_0)
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    immutable_list_1.find(immutable_list_1)


def test_case_19():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = module_0.ImmutableList(is_empty=immutable_list_0)
    var_0 = immutable_list_0.find(immutable_list_1)
    immutable_list_2 = module_0.ImmutableList(immutable_list_0)
    var_1 = immutable_list_2.__len__()
    immutable_list_0.filter(var_1)


def test_case_20():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = module_0.ImmutableList(immutable_list_0)
    immutable_list_1.find(immutable_list_0)


def test_case_21():
    int_0 = 0
    int_1 = 281
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.append(int_1)
    immutable_list_2 = module_0.ImmutableList(immutable_list_1, immutable_list_0, int_1)
    immutable_list_2.reduce(int_0, int_0)


def test_case_22():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = module_0.ImmutableList(is_empty=immutable_list_0)
    var_0 = immutable_list_0.find(immutable_list_1)
    immutable_list_2 = immutable_list_1.append(var_0)
    immutable_list_3 = immutable_list_0.unshift(immutable_list_0)
    bool_0 = immutable_list_2.__eq__(immutable_list_0)
    immutable_list_2.filter(var_0)
