# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.immutable_list as module_0


def test_case_0():
    bool_0 = True
    immutable_list_0 = module_0.ImmutableList(bool_0)
    bool_1 = immutable_list_0.__eq__(immutable_list_0)
    immutable_list_0.find(immutable_list_0)


def test_case_1():
    bool_0 = True
    immutable_list_0 = module_0.ImmutableList(bool_0)
    immutable_list_1 = immutable_list_0.append(immutable_list_0)
    bool_1 = immutable_list_0.__eq__(immutable_list_1)
    immutable_list_1.find(immutable_list_1)


def test_case_2():
    str_0 = "fiIOZct\\k7coO$)Xd"
    dict_0 = {}
    immutable_list_0 = module_0.ImmutableList(dict_0)
    immutable_list_1 = immutable_list_0.append(str_0)


def test_case_3():
    float_0 = -1146.524
    none_type_0 = None
    bool_0 = False
    immutable_list_0 = module_0.ImmutableList(tail=none_type_0, is_empty=bool_0)
    none_type_1 = None
    bool_1 = True
    immutable_list_1 = module_0.ImmutableList(is_empty=bool_1)
    immutable_list_2 = immutable_list_1.append(none_type_1)
    immutable_list_2.__add__(float_0)


def test_case_4():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.__len__()
    immutable_list_0.filter(var_0)


def test_case_5():
    bool_0 = True
    immutable_list_0 = module_0.ImmutableList(bool_0)
    var_0 = immutable_list_0.__len__()
    immutable_list_0.find(bool_0)


def test_case_6():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.to_list()
    var_1 = immutable_list_0.find(var_0)
    var_2 = immutable_list_0.find(var_1)


def test_case_7():
    bool_0 = False
    immutable_list_0 = module_0.ImmutableList(bool_0)
    immutable_list_1 = immutable_list_0.append(bool_0)
    str_0 = immutable_list_1.__str__()
    immutable_list_0.find(bool_0)


def test_case_8():
    none_type_0 = None
    none_type_1 = None
    immutable_list_0 = module_0.ImmutableList(none_type_1)
    immutable_list_0.map(none_type_0)


def test_case_9():
    none_type_0 = None
    bytes_0 = b"I\x8c\xa3\xa9\x88\xc4g}P#\x94G"
    immutable_list_0 = module_0.ImmutableList(tail=bytes_0)
    var_0 = immutable_list_0.__len__()
    bool_0 = False
    var_1 = immutable_list_0.reduce(var_0, bool_0)
    var_2 = immutable_list_0.reduce(var_1, var_1)
    immutable_list_0.map(none_type_0)


def test_case_10():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_0.filter(immutable_list_0)


def test_case_11():
    bool_0 = True
    immutable_list_0 = module_0.ImmutableList(is_empty=bool_0)
    bool_1 = immutable_list_0.__eq__(bool_0)
    none_type_0 = None
    immutable_list_1 = immutable_list_0.append(none_type_0)
    var_0 = immutable_list_0.to_list()
    var_1 = immutable_list_1.find(var_0)
    immutable_list_1.filter(var_1)


def test_case_12():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.reduce(immutable_list_0, immutable_list_0)
    var_1 = immutable_list_0.to_list()
    var_2 = immutable_list_0.__len__()
    var_0.map(var_0)


def test_case_13():
    immutable_list_0 = module_0.ImmutableList()


def test_case_14():
    immutable_list_0 = module_0.ImmutableList()
    str_0 = immutable_list_0.__str__()
    var_0 = immutable_list_0.to_list()
    immutable_list_1 = immutable_list_0.unshift(var_0)
    var_1 = immutable_list_1.__len__()
    var_2 = immutable_list_0.to_list()


def test_case_15():
    bool_0 = True
    immutable_list_0 = module_0.ImmutableList(bool_0)
    var_0 = immutable_list_0.unshift(immutable_list_0)
    immutable_list_0.find(bool_0)


def test_case_16():
    bool_0 = False
    immutable_list_0 = module_0.ImmutableList(bool_0)
    immutable_list_1 = immutable_list_0.append(immutable_list_0)
    immutable_list_2 = immutable_list_1.__add__(immutable_list_0)
    immutable_list_0.find(immutable_list_2)


def test_case_17():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.to_list()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    bool_0 = immutable_list_0.__eq__(immutable_list_1)
    bool_1 = immutable_list_0.__eq__(immutable_list_1)
    var_1 = immutable_list_0.find(bool_1)
    immutable_list_2 = module_0.ImmutableList()
    immutable_list_3 = immutable_list_2.unshift(var_1)
    immutable_list_0.filter(var_1)


def test_case_18():
    int_0 = -3086
    immutable_list_0 = module_0.ImmutableList(int_0, is_empty=int_0)
    immutable_list_0.find(int_0)


def test_case_19():
    bool_0 = True
    immutable_list_0 = module_0.ImmutableList(bool_0)
    immutable_list_1 = immutable_list_0.append(immutable_list_0)
    immutable_list_1.find(immutable_list_1)


def test_case_20():
    bool_0 = True
    immutable_list_0 = module_0.ImmutableList(is_empty=bool_0)
    bool_1 = immutable_list_0.__eq__(immutable_list_0)
    bool_2 = immutable_list_0.__eq__(bool_0)
    none_type_0 = None
    immutable_list_1 = immutable_list_0.append(none_type_0)
    var_0 = immutable_list_0.unshift(immutable_list_0)
    var_1 = immutable_list_0.find(var_0)
    immutable_list_2 = module_0.ImmutableList(is_empty=bool_1)
    str_0 = immutable_list_1.__str__()
    complex_0 = 3446.492 + 656j
    var_2 = immutable_list_2.reduce(complex_0, immutable_list_2)
    var_0.reduce(immutable_list_1, str_0)


def test_case_21():
    bool_0 = True
    bool_1 = True
    immutable_list_0 = module_0.ImmutableList(is_empty=bool_1)
    bool_2 = immutable_list_0.__eq__(bool_0)
    bool_3 = immutable_list_0.__eq__(immutable_list_0)
    immutable_list_1 = immutable_list_0.append(bool_3)
    var_0 = immutable_list_1.__len__()
    var_1 = immutable_list_0.find(bool_1)
    immutable_list_2 = module_0.ImmutableList(immutable_list_1)
    str_0 = var_1.__str__()
    immutable_list_2.reduce(var_1, var_1)
