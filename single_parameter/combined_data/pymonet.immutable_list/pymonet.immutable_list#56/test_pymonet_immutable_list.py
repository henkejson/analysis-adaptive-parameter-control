# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.immutable_list as module_0


def test_case_0():
    immutable_list_0 = module_0.ImmutableList()
    bool_0 = immutable_list_0.__eq__(immutable_list_0)
    immutable_list_0.filter(bool_0)


def test_case_1():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    var_0 = immutable_list_1.to_list()
    bool_0 = immutable_list_1.__eq__(var_0)
    immutable_list_1.find(immutable_list_1)


def test_case_2():
    bool_0 = True
    immutable_list_0 = module_0.ImmutableList(is_empty=bool_0)
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)


def test_case_3():
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(tail=none_type_0)
    immutable_list_1 = module_0.ImmutableList()
    bool_0 = True
    list_0 = [bool_0]
    immutable_list_2 = module_0.ImmutableList()
    bool_1 = immutable_list_2.__eq__(list_0)
    immutable_list_2.__add__(list_0)


def test_case_4():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    var_0 = immutable_list_1.__len__()
    immutable_list_1.find(var_0)


def test_case_5():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    var_0 = immutable_list_0.to_list()
    immutable_list_1.find(var_0)


def test_case_6():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    var_0 = immutable_list_1.to_list()
    immutable_list_1.find(immutable_list_1)


def test_case_7():
    bool_0 = True
    immutable_list_0 = module_0.ImmutableList(bool_0)
    str_0 = immutable_list_0.__str__()
    none_type_0 = None
    immutable_list_1 = immutable_list_0.unshift(none_type_0)
    immutable_list_2 = module_0.ImmutableList(tail=none_type_0)
    immutable_list_2.map(immutable_list_2)


def test_case_8():
    list_0 = []
    float_0 = 775.253
    immutable_list_0 = module_0.ImmutableList(float_0)
    none_type_0 = None
    immutable_list_1 = module_0.ImmutableList(none_type_0, none_type_0)
    immutable_list_2 = immutable_list_1.append(immutable_list_0)
    immutable_list_2.map(list_0)


def test_case_9():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_0.filter(immutable_list_0)


def test_case_10():
    list_0 = []
    immutable_list_0 = module_0.ImmutableList(tail=list_0, is_empty=list_0)
    immutable_list_0.filter(list_0)


def test_case_11():
    immutable_list_0 = module_0.ImmutableList()
    bool_0 = immutable_list_0.find(immutable_list_0)
    immutable_list_0.filter(bool_0)


def test_case_12():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.reduce(immutable_list_0, immutable_list_0)
    immutable_list_1 = var_0.unshift(var_0)
    immutable_list_1.find(var_0)


def test_case_13():
    bytes_0 = b"\x04\xf5\xca\xcfj\xfa\xa2\x84\xd2\x88L:h\xfd"
    immutable_list_0 = module_0.ImmutableList(bytes_0)
    var_0 = immutable_list_0.__len__()
    immutable_list_0.reduce(bytes_0, bytes_0)


def test_case_14():
    immutable_list_0 = module_0.ImmutableList()


def test_case_15():
    immutable_list_0 = module_0.ImmutableList()
    str_0 = immutable_list_0.__str__()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    immutable_list_1.find(immutable_list_0)


def test_case_16():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.append(immutable_list_0)
    var_0 = immutable_list_0.find(immutable_list_0)
    immutable_list_0.filter(var_0)


def test_case_17():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    var_0 = immutable_list_1.__add__(immutable_list_1)
    immutable_list_1.find(var_0)


def test_case_18():
    bool_0 = True
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(bool_0, none_type_0, bool_0)
    var_0 = immutable_list_0.__len__()
    immutable_list_0.find(var_0)


def test_case_19():
    int_0 = -2512
    none_type_0 = None
    bool_0 = True
    immutable_list_0 = module_0.ImmutableList(is_empty=bool_0)
    immutable_list_1 = immutable_list_0.unshift(none_type_0)
    immutable_list_2 = module_0.ImmutableList(is_empty=int_0)
    immutable_list_3 = immutable_list_2.unshift(int_0)
    immutable_list_4 = immutable_list_2.__add__(immutable_list_2)
    bool_1 = True
    bool_2 = immutable_list_2.__eq__(immutable_list_0)
    immutable_list_5 = immutable_list_4.append(bool_1)
    bool_3 = immutable_list_2.__eq__(immutable_list_2)
    immutable_list_6 = immutable_list_3.__add__(immutable_list_4)
    str_0 = immutable_list_2.__str__()
    var_0 = immutable_list_2.__len__()
    immutable_list_3.reduce(var_0, immutable_list_2)


def test_case_20():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    immutable_list_1.find(immutable_list_1)


def test_case_21():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    var_0 = immutable_list_0.append(immutable_list_1)
    bool_0 = immutable_list_1.__eq__(var_0)
    immutable_list_1.find(immutable_list_1)


def test_case_22():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.__add__(immutable_list_0)
    immutable_list_2 = module_0.ImmutableList(immutable_list_0)
    bool_0 = immutable_list_0.__eq__(immutable_list_1)
    immutable_list_2.find(immutable_list_2)


def test_case_23():
    bool_0 = True
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(bool_0, none_type_0, bool_0)
    immutable_list_0.find(bool_0)
