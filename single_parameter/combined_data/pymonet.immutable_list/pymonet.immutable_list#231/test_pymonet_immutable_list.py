# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.immutable_list as module_0


def test_case_0():
    immutable_list_0 = module_0.ImmutableList()
    bool_0 = immutable_list_0.__eq__(immutable_list_0)
    str_0 = immutable_list_0.__str__()


def test_case_1():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    bool_0 = immutable_list_0.__eq__(immutable_list_1)
    none_type_0 = None
    immutable_list_1.find(none_type_0)


def test_case_2():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    immutable_list_1.find(immutable_list_0)


def test_case_3():
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(none_type_0)
    var_0 = immutable_list_0.to_list()
    str_0 = immutable_list_0.__str__()
    immutable_list_1 = immutable_list_0.append(var_0)
    immutable_list_1.__add__(var_0)


def test_case_4():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.__len__()


def test_case_5():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    var_0 = immutable_list_1.__len__()
    immutable_list_1.find(immutable_list_0)


def test_case_6():
    immutable_list_0 = module_0.ImmutableList()
    str_0 = immutable_list_0.__str__()


def test_case_7():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    str_0 = immutable_list_1.__str__()
    immutable_list_1.find(immutable_list_0)


def test_case_8():
    none_type_0 = None
    none_type_1 = None
    immutable_list_0 = module_0.ImmutableList(tail=none_type_1)
    var_0 = immutable_list_0.reduce(none_type_0, none_type_0)
    bool_0 = True
    immutable_list_1 = module_0.ImmutableList(is_empty=bool_0)
    immutable_list_1.map(var_0)


def test_case_9():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_0.filter(immutable_list_0)


def test_case_10():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    immutable_list_2 = immutable_list_0.append(immutable_list_0)
    var_0 = immutable_list_1.__len__()
    immutable_list_3 = immutable_list_0.__add__(immutable_list_2)
    var_1 = immutable_list_2.find(immutable_list_0)
    immutable_list_4 = immutable_list_3.append(var_0)
    immutable_list_3.filter(immutable_list_1)


def test_case_11():
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(none_type_0, none_type_0, none_type_0)
    bool_0 = immutable_list_0.__eq__(immutable_list_0)
    bool_1 = immutable_list_0.__eq__(immutable_list_0)
    var_0 = immutable_list_0.find(immutable_list_0)


def test_case_12():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    var_0 = immutable_list_0.reduce(immutable_list_1, immutable_list_1)
    var_0.find(immutable_list_0)


def test_case_13():
    immutable_list_0 = module_0.ImmutableList()


def test_case_14():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    var_0 = immutable_list_0.append(immutable_list_1)
    immutable_list_1.find(immutable_list_0)


def test_case_15():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    var_0 = immutable_list_0.reduce(immutable_list_1, immutable_list_1)
    var_1 = var_0.append(var_0)
    var_0.find(immutable_list_0)


def test_case_16():
    str_0 = "^EyE5P]!}(~0ny\r9[miR"
    immutable_list_0 = module_0.ImmutableList(is_empty=str_0)
    immutable_list_1 = module_0.ImmutableList()
    immutable_list_2 = immutable_list_1.__add__(immutable_list_0)
    immutable_list_3 = immutable_list_2.__add__(immutable_list_0)
    immutable_list_4 = immutable_list_1.unshift(str_0)
    bool_0 = immutable_list_4.__eq__(immutable_list_4)
    immutable_list_5 = module_0.ImmutableList()
    var_0 = immutable_list_4.to_list()
    immutable_list_6 = immutable_list_1.__add__(immutable_list_5)
    immutable_list_4.reduce(immutable_list_4, immutable_list_5)


def test_case_17():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = module_0.ImmutableList(
        immutable_list_0, is_empty=immutable_list_0
    )
    immutable_list_2 = immutable_list_1.unshift(immutable_list_1)
    immutable_list_1.find(immutable_list_0)


def test_case_18():
    bool_0 = True
    immutable_list_0 = module_0.ImmutableList(bool_0, is_empty=bool_0)
    immutable_list_1 = immutable_list_0.__add__(immutable_list_0)
    var_0 = immutable_list_0.__len__()
    var_1 = immutable_list_0.__len__()
    bool_1 = immutable_list_1.__eq__(immutable_list_0)
    immutable_list_0.find(var_0)


def test_case_19():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.append(immutable_list_0)
    bool_0 = False
    immutable_list_2 = immutable_list_0.__add__(immutable_list_0)
    var_0 = immutable_list_1.find(bool_0)
    immutable_list_1.map(immutable_list_0)


def test_case_20():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = module_0.ImmutableList(
        immutable_list_0, is_empty=immutable_list_0
    )
    immutable_list_2 = immutable_list_0.unshift(immutable_list_1)
    var_0 = immutable_list_1.__len__()
    immutable_list_3 = immutable_list_1.unshift(immutable_list_1)
    immutable_list_1.find(immutable_list_0)


def test_case_21():
    bool_0 = True
    tuple_0 = (bool_0,)
    immutable_list_0 = module_0.ImmutableList(tuple_0)
    none_type_0 = None
    immutable_list_0.reduce(none_type_0, bool_0)
