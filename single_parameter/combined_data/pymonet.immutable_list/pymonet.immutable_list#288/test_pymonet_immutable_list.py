# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.immutable_list as module_0


def test_case_0():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = module_0.ImmutableList()
    bool_0 = immutable_list_0.__eq__(immutable_list_0)
    bool_1 = immutable_list_0.__eq__(immutable_list_0)
    var_0 = immutable_list_0.to_list()


def test_case_1():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = module_0.ImmutableList()
    var_0 = immutable_list_1.__len__()
    bool_0 = immutable_list_0.__eq__(var_0)
    immutable_list_1.filter(var_0)


def test_case_2():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    immutable_list_1.filter(immutable_list_0)


def test_case_3():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = module_0.ImmutableList()
    str_0 = immutable_list_0.__str__()
    var_0 = immutable_list_1.to_list()
    set_0 = set()
    none_type_0 = None
    var_1 = immutable_list_0.reduce(set_0, none_type_0)
    immutable_list_0.__add__(var_1)


def test_case_4():
    immutable_list_0 = module_0.ImmutableList()
    str_0 = immutable_list_0.__str__()
    var_0 = immutable_list_0.__len__()
    var_1 = immutable_list_0.find(var_0)


def test_case_5():
    int_0 = 0
    immutable_list_0 = module_0.ImmutableList(int_0, is_empty=int_0)
    var_0 = immutable_list_0.__len__()
    immutable_list_0.find(int_0)


def test_case_6():
    immutable_list_0 = module_0.ImmutableList()
    bool_0 = immutable_list_0.find(immutable_list_0)
    immutable_list_1 = immutable_list_0.__add__(immutable_list_0)
    bool_1 = immutable_list_1.__eq__(bool_0)
    var_0 = immutable_list_1.to_list()
    var_1 = immutable_list_1.find(var_0)
    immutable_list_0.filter(var_1)


def test_case_7():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_0.map(immutable_list_0)


def test_case_8():
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(none_type_0)
    immutable_list_2 = module_0.ImmutableList()
    immutable_list_1.map(immutable_list_1)


def test_case_9():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_0.filter(immutable_list_0)


def test_case_10():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.find(immutable_list_0)
    immutable_list_0.filter(immutable_list_0)


def test_case_11():
    immutable_list_0 = module_0.ImmutableList()
    none_type_0 = None
    var_0 = immutable_list_0.reduce(none_type_0, immutable_list_0)
    var_0.filter(immutable_list_0)


def test_case_12():
    bool_0 = False
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(bool_0)
    immutable_list_2 = immutable_list_1.unshift(bool_0)
    immutable_list_3 = immutable_list_2.unshift(bool_0)
    immutable_list_3.reduce(none_type_0, none_type_0)


def test_case_13():
    immutable_list_0 = module_0.ImmutableList()


def test_case_14():
    int_0 = 6
    immutable_list_0 = module_0.ImmutableList(int_0, is_empty=int_0)
    immutable_list_1 = immutable_list_0.append(int_0)
    immutable_list_0.find(int_0)


def test_case_15():
    bool_0 = False
    immutable_list_0 = module_0.ImmutableList(is_empty=bool_0)
    bytes_0 = b"_:\x98H\xaba\xeeab+\xcf5l\xa0\xdc:"
    immutable_list_1 = module_0.ImmutableList()
    immutable_list_2 = immutable_list_1.unshift(bytes_0)
    immutable_list_3 = immutable_list_2.append(immutable_list_0)


def test_case_16():
    float_0 = 872.4420945990721
    immutable_list_0 = module_0.ImmutableList(float_0)
    immutable_list_0.reduce(float_0, immutable_list_0)


def test_case_17():
    int_0 = 0
    immutable_list_0 = module_0.ImmutableList(int_0, is_empty=int_0)
    immutable_list_0.find(int_0)


def test_case_18():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    bool_0 = immutable_list_0.__eq__(immutable_list_1)
    var_0 = immutable_list_1.__len__()
    immutable_list_1.find(immutable_list_1)


def test_case_19():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    var_0 = immutable_list_1.__len__()
    immutable_list_1.find(immutable_list_1)


def test_case_20():
    int_0 = -802
    immutable_list_0 = module_0.ImmutableList(int_0)
    immutable_list_1 = immutable_list_0.append(immutable_list_0)
    bool_0 = immutable_list_0.__eq__(immutable_list_1)
    immutable_list_0.find(bool_0)


def test_case_21():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    immutable_list_1.find(immutable_list_1)
