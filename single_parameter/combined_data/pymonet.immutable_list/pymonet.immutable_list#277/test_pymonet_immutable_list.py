# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.immutable_list as module_0


def test_case_0():
    immutable_list_0 = module_0.ImmutableList()
    bool_0 = immutable_list_0.__eq__(immutable_list_0)
    immutable_list_1 = module_0.ImmutableList(immutable_list_0)
    immutable_list_1.find(bool_0)


def test_case_1():
    int_0 = 1279
    immutable_list_0 = module_0.ImmutableList()
    bool_0 = immutable_list_0.__eq__(int_0)
    immutable_list_1 = immutable_list_0.unshift(int_0)
    immutable_list_2 = module_0.ImmutableList(immutable_list_1)
    immutable_list_2.find(int_0)


def test_case_2():
    int_0 = 1279
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(int_0)
    immutable_list_1.find(int_0)


def test_case_3():
    float_0 = -504.6632
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(tail=none_type_0, is_empty=none_type_0)
    immutable_list_0.__add__(float_0)


def test_case_4():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.__len__()
    bool_0 = var_0.__eq__(immutable_list_0)
    immutable_list_0.filter(var_0)


def test_case_5():
    bool_0 = True
    immutable_list_0 = module_0.ImmutableList(bool_0)
    var_0 = immutable_list_0.__len__()
    immutable_list_0.filter(var_0)


def test_case_6():
    immutable_list_0 = module_0.ImmutableList()
    str_0 = immutable_list_0.__str__()


def test_case_7():
    bool_0 = True
    bool_1 = True
    immutable_list_0 = module_0.ImmutableList(tail=bool_0, is_empty=bool_1)
    immutable_list_0.__str__()


def test_case_8():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_0.map(immutable_list_0)


def test_case_9():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_0.filter(immutable_list_0)


def test_case_10():
    none_type_0 = None
    bool_0 = False
    immutable_list_0 = module_0.ImmutableList(tail=bool_0)
    immutable_list_0.filter(none_type_0)


def test_case_11():
    int_0 = 1279
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(int_0)
    var_0 = immutable_list_0.find(immutable_list_1)
    bool_0 = immutable_list_1.__eq__(immutable_list_0)
    immutable_list_1.find(int_0)


def test_case_12():
    int_0 = -2264
    immutable_list_0 = module_0.ImmutableList(int_0, is_empty=int_0)
    immutable_list_0.find(int_0)


def test_case_13():
    bool_0 = True
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(is_empty=none_type_0)
    var_0 = immutable_list_0.reduce(bool_0, none_type_0)
    var_0.append(bool_0)


def test_case_14():
    immutable_list_0 = module_0.ImmutableList()


def test_case_15():
    int_0 = 1279
    immutable_list_0 = module_0.ImmutableList(int_0)
    immutable_list_1 = immutable_list_0.append(int_0)
    immutable_list_0.find(int_0)


def test_case_16():
    int_0 = 1279
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(int_0)
    immutable_list_2 = immutable_list_1.__add__(immutable_list_1)
    immutable_list_2.find(int_0)


def test_case_17():
    bool_0 = True
    immutable_list_0 = module_0.ImmutableList(bool_0, bool_0, bool_0)
    immutable_list_0.find(bool_0)


def test_case_18():
    int_0 = 1279
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(int_0)
    bool_0 = immutable_list_1.__eq__(immutable_list_0)
    immutable_list_1.find(int_0)


def test_case_19():
    int_0 = -1276
    immutable_list_0 = module_0.ImmutableList(int_0, int_0)
    immutable_list_0.map(immutable_list_0)


def test_case_20():
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(tail=none_type_0)
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    var_0 = immutable_list_0.find(none_type_0)
    str_0 = immutable_list_1.__str__()
    bool_0 = immutable_list_1.__eq__(immutable_list_1)
    immutable_list_1.reduce(none_type_0, none_type_0)


def test_case_21():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.append(immutable_list_0)
    bool_0 = immutable_list_1.__eq__(immutable_list_0)
    none_type_0 = None
    immutable_list_1.__add__(none_type_0)


def test_case_22():
    int_0 = 1279
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(int_0)
    immutable_list_2 = immutable_list_1.append(immutable_list_1)
    var_0 = immutable_list_1.__len__()
    immutable_list_2.find(int_0)


def test_case_23():
    bytes_0 = b"\xefdN\xd1\x1di\x1a\xb4\xf4\x15\xc5\xf3`m\tC\x07\\"
    immutable_list_0 = module_0.ImmutableList(bytes_0)
    bool_0 = immutable_list_0.__eq__(bytes_0)
    immutable_list_0.reduce(bytes_0, bytes_0)
