# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.immutable_list as module_0


def test_case_0():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.append(immutable_list_0)
    var_0 = immutable_list_0.__add__(immutable_list_0)
    bool_0 = immutable_list_0.__eq__(immutable_list_1)


def test_case_1():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.append(immutable_list_0)
    immutable_list_1.filter(immutable_list_0)


def test_case_2():
    float_0 = 2278.160963
    float_1 = 1128.46894
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(tail=none_type_0)
    immutable_list_1 = immutable_list_0.append(float_1)
    immutable_list_1.__add__(float_0)


def test_case_3():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.__len__()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    immutable_list_1.find(immutable_list_1)


def test_case_4():
    float_0 = 1707.36891
    immutable_list_0 = module_0.ImmutableList(float_0)
    var_0 = immutable_list_0.__len__()
    immutable_list_0.find(var_0)


def test_case_5():
    immutable_list_0 = module_0.ImmutableList()
    str_0 = immutable_list_0.__str__()
    immutable_list_1 = module_0.ImmutableList()


def test_case_6():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.append(immutable_list_0)
    var_0 = immutable_list_1.to_list()
    var_1 = immutable_list_0.find(immutable_list_0)
    immutable_list_1.filter(var_0)


def test_case_7():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.find(immutable_list_0)
    var_0 = immutable_list_0.to_list()
    immutable_list_0.map(immutable_list_1)


def test_case_8():
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(tail=none_type_0)
    immutable_list_1 = immutable_list_0.append(none_type_0)
    immutable_list_2 = immutable_list_0.append(none_type_0)
    immutable_list_2.map(none_type_0)


def test_case_9():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_0.filter(immutable_list_0)


def test_case_10():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.find(immutable_list_0)
    immutable_list_0.filter(var_0)


def test_case_11():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.reduce(immutable_list_0, immutable_list_0)
    var_1 = var_0.find(var_0)
    var_0.filter(immutable_list_0)


def test_case_12():
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(none_type_0)
    int_0 = 435
    set_0 = set()
    none_type_1 = None
    immutable_list_1 = module_0.ImmutableList(set_0, none_type_1)
    immutable_list_1.reduce(int_0, int_0)


def test_case_13():
    immutable_list_0 = module_0.ImmutableList()


def test_case_14():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.unshift(immutable_list_0)
    var_0.find(var_0)


def test_case_15():
    int_0 = 1064
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(none_type_0)
    immutable_list_2 = immutable_list_1.append(int_0)


def test_case_16():
    bytes_0 = b"\xf5T=4~\x82\x90\xefB)\xcb"
    immutable_list_0 = module_0.ImmutableList()
    bytes_1 = b"\xf4"
    bool_0 = False
    immutable_list_1 = module_0.ImmutableList(bool_0, is_empty=bool_0)
    immutable_list_2 = immutable_list_1.append(bytes_1)
    immutable_list_3 = immutable_list_2.__add__(immutable_list_0)
    immutable_list_3.reduce(bytes_0, bytes_0)


def test_case_17():
    float_0 = 1344.0
    immutable_list_0 = module_0.ImmutableList(float_0)
    bool_0 = immutable_list_0.__eq__(immutable_list_0)
    str_0 = "\n\nFJ%O\x0b:E^C"
    immutable_list_0.find(str_0)


def test_case_18():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    var_0 = immutable_list_1.__len__()
    immutable_list_1.find(var_0)


def test_case_19():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    bool_0 = immutable_list_0.__eq__(immutable_list_1)
    immutable_list_1.find(bool_0)


def test_case_20():
    float_0 = 1344.0
    immutable_list_0 = module_0.ImmutableList(float_0)
    str_0 = "\n\nFJ%O\x0b:E^C"
    immutable_list_0.find(str_0)
