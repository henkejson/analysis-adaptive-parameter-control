# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.immutable_list as module_0


def test_case_0():
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(none_type_0, is_empty=none_type_0)
    var_0 = immutable_list_0.to_list()
    bool_0 = immutable_list_0.__eq__(immutable_list_0)
    immutable_list_1 = immutable_list_0.unshift(none_type_0)
    var_1 = immutable_list_0.to_list()


def test_case_1():
    bool_0 = False
    immutable_list_0 = module_0.ImmutableList(bool_0)
    bool_1 = immutable_list_0.__eq__(bool_0)
    immutable_list_0.find(immutable_list_0)


def test_case_2():
    bool_0 = True
    immutable_list_0 = module_0.ImmutableList(bool_0)
    immutable_list_1 = immutable_list_0.append(bool_0)
    immutable_list_0.find(immutable_list_1)


def test_case_3():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.reduce(immutable_list_0, immutable_list_0)
    immutable_list_1 = immutable_list_0.__add__(immutable_list_0)
    var_1 = immutable_list_0.__len__()
    immutable_list_1.__add__(var_1)


def test_case_4():
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(is_empty=none_type_0)
    var_0 = immutable_list_0.__len__()
    var_0.to_list()


def test_case_5():
    bool_0 = False
    immutable_list_0 = module_0.ImmutableList(bool_0)
    immutable_list_1 = immutable_list_0.__add__(immutable_list_0)
    var_0 = immutable_list_1.__len__()
    immutable_list_1.find(immutable_list_0)


def test_case_6():
    bool_0 = False
    immutable_list_0 = module_0.ImmutableList(tail=bool_0)
    var_0 = immutable_list_0.find(bool_0)
    var_1 = immutable_list_0.reduce(var_0, var_0)
    immutable_list_0.to_list()


def test_case_7():
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(tail=none_type_0)
    immutable_list_1 = immutable_list_0.unshift(none_type_0)
    var_0 = immutable_list_0.__len__()
    immutable_list_0.map(var_0)


def test_case_8():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_0.filter(immutable_list_0)


def test_case_9():
    bool_0 = True
    bytes_0 = b"\xae\x0b\xd1\xb4\x84\xab{e"
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = module_0.ImmutableList(tail=bytes_0)
    immutable_list_2 = immutable_list_0.__add__(immutable_list_0)
    immutable_list_3 = immutable_list_0.__add__(immutable_list_2)
    immutable_list_4 = immutable_list_3.unshift(immutable_list_1)
    immutable_list_5 = immutable_list_2.append(immutable_list_4)
    var_0 = immutable_list_0.__len__()
    immutable_list_1.filter(bool_0)


def test_case_10():
    immutable_list_0 = module_0.ImmutableList()
    none_type_0 = None
    var_0 = immutable_list_0.find(none_type_0)


def test_case_11():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.reduce(immutable_list_0, immutable_list_0)
    immutable_list_1 = immutable_list_0.__add__(immutable_list_0)
    var_1 = immutable_list_0.__len__()
    var_1.unshift(immutable_list_0)


def test_case_12():
    immutable_list_0 = module_0.ImmutableList()


def test_case_13():
    float_0 = 1590.0
    dict_0 = {float_0: float_0, float_0: float_0}
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(dict_0, none_type_0)
    immutable_list_1 = immutable_list_0.unshift(float_0)
    str_0 = immutable_list_0.__str__()


def test_case_14():
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    immutable_list_1.find(none_type_0)


def test_case_15():
    bool_0 = False
    immutable_list_0 = module_0.ImmutableList(bool_0)
    immutable_list_1 = immutable_list_0.__add__(immutable_list_0)
    var_0 = immutable_list_1.append(immutable_list_1)
    immutable_list_0.find(immutable_list_0)


def test_case_16():
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(none_type_0)
    str_0 = immutable_list_0.__str__()
    immutable_list_1 = module_0.ImmutableList(tail=str_0)
    var_0 = immutable_list_0.to_list()
    immutable_list_2 = immutable_list_0.unshift(immutable_list_0)
    var_1 = immutable_list_1.find(var_0)
    immutable_list_3 = immutable_list_2.append(var_1)
    immutable_list_3.map(none_type_0)


def test_case_17():
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(none_type_0)
    immutable_list_1 = immutable_list_0.unshift(none_type_0)
    immutable_list_2 = immutable_list_1.unshift(none_type_0)
    var_0 = immutable_list_1.reduce(immutable_list_0, immutable_list_1)
    bool_0 = immutable_list_2.__eq__(immutable_list_1)
    var_1 = immutable_list_0.__len__()
    var_2 = var_0.__len__()
    var_2.unshift(immutable_list_2)


def test_case_18():
    bool_0 = True
    immutable_list_0 = module_0.ImmutableList(bool_0)
    immutable_list_0.find(immutable_list_0)


def test_case_19():
    bool_0 = False
    immutable_list_0 = module_0.ImmutableList(bool_0)
    var_0 = immutable_list_0.__len__()
    var_1 = immutable_list_0.__len__()
    bool_1 = immutable_list_0.__eq__(bool_0)
    immutable_list_0.reduce(var_1, bool_1)


def test_case_20():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = module_0.ImmutableList(immutable_list_0)
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    var_1 = var_0.append(var_0)
    immutable_list_2 = var_1.__add__(var_1)
    immutable_list_1.reduce(immutable_list_2, immutable_list_2)


def test_case_21():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    bool_0 = immutable_list_1.__eq__(immutable_list_0)
    var_0 = immutable_list_0.__len__()
    none_type_0 = None
    immutable_list_2 = module_0.ImmutableList()
    var_1 = immutable_list_2.reduce(immutable_list_2, none_type_0)
    var_2 = immutable_list_2.reduce(var_1, none_type_0)
    bool_1 = False
    immutable_list_3 = module_0.ImmutableList(tail=none_type_0, is_empty=bool_1)
    immutable_list_1.find(immutable_list_2)
