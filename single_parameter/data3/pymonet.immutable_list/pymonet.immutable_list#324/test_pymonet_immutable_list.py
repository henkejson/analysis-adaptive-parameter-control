# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.immutable_list as module_0


def test_case_0():
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(is_empty=none_type_0)
    bool_0 = immutable_list_0.__eq__(immutable_list_0)
    immutable_list_1 = immutable_list_0.unshift(bool_0)
    immutable_list_1.find(none_type_0)


def test_case_1():
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(is_empty=none_type_0)
    immutable_list_1 = module_0.ImmutableList(immutable_list_0)
    bool_0 = immutable_list_0.__eq__(immutable_list_1)
    immutable_list_1.find(none_type_0)


def test_case_2():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.__add__(immutable_list_0)


def test_case_3():
    bytes_0 = b"W\xa1\xfc|\xd0\x12N\x85"
    bool_0 = True
    immutable_list_0 = module_0.ImmutableList(bytes_0, is_empty=bool_0)
    var_0 = immutable_list_0.__len__()
    immutable_list_1 = immutable_list_0.append(immutable_list_0)
    immutable_list_2 = module_0.ImmutableList(bytes_0)
    immutable_list_2.__add__(var_0)


def test_case_4():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    var_0 = immutable_list_1.__len__()
    immutable_list_1.find(var_0)


def test_case_5():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.to_list()
    immutable_list_0.filter(var_0)


def test_case_6():
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(is_empty=none_type_0)
    bool_0 = immutable_list_0.__eq__(immutable_list_0)
    immutable_list_1 = immutable_list_0.unshift(bool_0)
    var_0 = immutable_list_1.to_list()
    immutable_list_1.find(none_type_0)


def test_case_7():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.find(immutable_list_0)
    var_1 = immutable_list_0.to_list()
    str_0 = immutable_list_0.__str__()
    immutable_list_0.map(immutable_list_0)


def test_case_8():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_0.filter(immutable_list_0)


def test_case_9():
    bool_0 = False
    immutable_list_0 = module_0.ImmutableList(tail=bool_0)
    immutable_list_0.filter(bool_0)


def test_case_10():
    str_0 = " +QkFhpI;."
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0}
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.find(dict_0)


def test_case_11():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    immutable_list_1.find(immutable_list_1)


def test_case_12():
    str_0 = "\x0c"
    bool_0 = True
    bool_1 = True
    immutable_list_0 = module_0.ImmutableList(is_empty=bool_1)
    var_0 = immutable_list_0.reduce(str_0, bool_0)


def test_case_13():
    float_0 = -2228.7
    int_0 = 1
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(int_0, none_type_0, none_type_0)
    immutable_list_0.reduce(none_type_0, float_0)


def test_case_14():
    immutable_list_0 = module_0.ImmutableList()


def test_case_15():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    var_0 = immutable_list_1.__len__()
    str_0 = immutable_list_0.__str__()
    immutable_list_1.find(var_0)


def test_case_16():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.append(immutable_list_0)
    immutable_list_0.filter(var_0)


def test_case_17():
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(is_empty=none_type_0)
    bool_0 = immutable_list_0.__eq__(immutable_list_0)
    immutable_list_1 = immutable_list_0.unshift(bool_0)
    var_0 = immutable_list_1.__add__(immutable_list_0)
    immutable_list_1.find(none_type_0)


def test_case_18():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = module_0.ImmutableList(immutable_list_0)
    immutable_list_2 = immutable_list_1.__len__()
    immutable_list_1.find(immutable_list_1)


def test_case_19():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = module_0.ImmutableList(immutable_list_0)
    immutable_list_1.find(immutable_list_1)


def test_case_20():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.reduce(immutable_list_0, immutable_list_0)
    var_1 = var_0.reduce(var_0, var_0)
    str_0 = var_0.__str__()
    immutable_list_1 = var_0.unshift(var_0)
    str_1 = immutable_list_0.__str__()
    immutable_list_1.map(var_1)


def test_case_21():
    bool_0 = False
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(is_empty=none_type_0)
    var_0 = immutable_list_0.__len__()
    var_1 = immutable_list_0.reduce(var_0, none_type_0)
    immutable_list_1 = immutable_list_0.unshift(bool_0)
    immutable_list_1.reduce(var_1, none_type_0)


def test_case_22():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.append(immutable_list_0)
    bool_0 = immutable_list_1.__eq__(immutable_list_0)
    var_0 = immutable_list_1.reduce(immutable_list_1, immutable_list_1)
    var_1 = immutable_list_1.find(var_0)
    var_0.filter(var_0)
