# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.immutable_list as module_0


def test_case_0():
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(tail=none_type_0, is_empty=none_type_0)
    bool_0 = immutable_list_0.__eq__(immutable_list_0)
    var_0 = immutable_list_0.find(immutable_list_0)
    var_0.filter(var_0)


def test_case_1():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    bool_0 = immutable_list_0.__eq__(immutable_list_1)
    immutable_list_1.find(immutable_list_1)


def test_case_2():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    immutable_list_1.find(immutable_list_1)


def test_case_3():
    bool_0 = False
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(bool_0, none_type_0)
    bool_1 = immutable_list_0.__eq__(bool_0)
    var_0 = immutable_list_0.__len__()
    immutable_list_0.__add__(bool_0)


def test_case_4():
    bool_0 = False
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.__len__()
    var_0.append(bool_0)


def test_case_5():
    int_0 = 457
    immutable_list_0 = module_0.ImmutableList(int_0)
    var_0 = immutable_list_0.__len__()
    immutable_list_1 = var_0.__add__(var_0)
    immutable_list_0.find(var_0)


def test_case_6():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.to_list()
    var_1 = immutable_list_0.find(var_0)
    immutable_list_0.filter(var_1)


def test_case_7():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    var_0 = immutable_list_1.to_list()
    immutable_list_1.find(var_0)


def test_case_8():
    none_type_0 = None
    bool_0 = True
    immutable_list_0 = module_0.ImmutableList(is_empty=bool_0)
    immutable_list_0.map(none_type_0)


def test_case_9():
    bool_0 = False
    bool_1 = True
    immutable_list_0 = module_0.ImmutableList(tail=bool_0, is_empty=bool_1)
    immutable_list_0.map(immutable_list_0)


def test_case_10():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_0.filter(immutable_list_0)


def test_case_11():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.to_list()
    var_1 = immutable_list_0.find(var_0)
    var_2 = immutable_list_0.find(var_1)
    immutable_list_1 = immutable_list_0.__add__(immutable_list_0)
    immutable_list_2 = immutable_list_1.unshift(immutable_list_1)
    bool_0 = immutable_list_0.__eq__(immutable_list_1)
    immutable_list_2.filter(immutable_list_2)


def test_case_12():
    bool_0 = False
    immutable_list_0 = module_0.ImmutableList(bool_0)
    immutable_list_0.find(bool_0)


def test_case_13():
    float_0 = -3946.9
    list_0 = [float_0, float_0]
    bool_0 = True
    immutable_list_0 = module_0.ImmutableList(list_0, is_empty=bool_0)
    immutable_list_1 = module_0.ImmutableList()
    var_0 = immutable_list_1.reduce(list_0, bool_0)
    immutable_list_0.find(immutable_list_1)


def test_case_14():
    immutable_list_0 = module_0.ImmutableList()


def test_case_15():
    none_type_0 = None
    bool_0 = False
    immutable_list_0 = module_0.ImmutableList(bool_0)
    str_0 = immutable_list_0.__str__()
    immutable_list_0.find(none_type_0)


def test_case_16():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.append(immutable_list_0)
    var_0 = immutable_list_0.to_list()
    var_1 = immutable_list_1.find(var_0)
    immutable_list_0.filter(var_1)


def test_case_17():
    int_0 = 457
    immutable_list_0 = module_0.ImmutableList(int_0)
    var_0 = immutable_list_0.unshift(immutable_list_0)
    immutable_list_1 = var_0.__add__(var_0)
    immutable_list_0.find(var_0)


def test_case_18():
    int_0 = -522
    list_0 = [int_0, int_0, int_0, int_0]
    int_1 = 872
    tuple_0 = (list_0, int_1)
    immutable_list_0 = module_0.ImmutableList(tuple_0, tuple_0)
    var_0 = immutable_list_0.__len__()


def test_case_19():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.__add__(immutable_list_0)
    immutable_list_2 = immutable_list_1.unshift(immutable_list_1)
    bool_0 = immutable_list_0.__eq__(immutable_list_1)
    immutable_list_2.find(bool_0)


def test_case_20():
    str_0 = "0\ng\nRi\x0bapt"
    immutable_list_0 = module_0.ImmutableList(str_0)
    immutable_list_0.reduce(immutable_list_0, str_0)


def test_case_21():
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(tail=none_type_0)
    bool_0 = immutable_list_0.__eq__(immutable_list_0)
    immutable_list_1 = module_0.ImmutableList()
    immutable_list_2 = immutable_list_0.__add__(immutable_list_0)
    bool_1 = False
    immutable_list_3 = module_0.ImmutableList(
        none_type_0, immutable_list_0, none_type_0
    )
    immutable_list_4 = module_0.ImmutableList(is_empty=immutable_list_2)
    immutable_list_5 = immutable_list_2.unshift(immutable_list_2)
    immutable_list_6 = immutable_list_5.append(none_type_0)
    bool_2 = immutable_list_5.__eq__(immutable_list_3)
    var_0 = immutable_list_6.__len__()
    var_1 = immutable_list_0.find(var_0)
    bool_3 = immutable_list_3.__eq__(bool_1)
    immutable_list_5.reduce(var_1, none_type_0)
