# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.immutable_list as module_0


def test_case_0():
    immutable_list_0 = module_0.ImmutableList()
    bool_0 = immutable_list_0.__eq__(immutable_list_0)
    immutable_list_1 = immutable_list_0.append(immutable_list_0)
    var_0 = immutable_list_1.find(immutable_list_1)
    str_0 = immutable_list_0.__str__()
    bool_1 = immutable_list_0.__eq__(immutable_list_0)


def test_case_1():
    bool_0 = False
    set_0 = {bool_0}
    immutable_list_0 = module_0.ImmutableList()
    bool_1 = immutable_list_0.__eq__(set_0)


def test_case_2():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.append(immutable_list_0)


def test_case_3():
    bytes_0 = b"\x9d\r\x12"
    immutable_list_0 = module_0.ImmutableList(tail=bytes_0)
    immutable_list_1 = module_0.ImmutableList(bytes_0, is_empty=bytes_0)
    immutable_list_2 = immutable_list_0.unshift(immutable_list_1)
    var_0 = immutable_list_1.to_list()
    immutable_list_1.__add__(var_0)


def test_case_4():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.__len__()
    immutable_list_0.filter(var_0)


def test_case_5():
    bool_0 = True
    immutable_list_0 = module_0.ImmutableList(bool_0, is_empty=bool_0)
    var_0 = immutable_list_0.__len__()
    immutable_list_0.find(bool_0)


def test_case_6():
    bool_0 = False
    immutable_list_0 = module_0.ImmutableList(is_empty=bool_0)
    str_0 = immutable_list_0.__str__()
    var_0 = immutable_list_0.find(bool_0)
    immutable_list_1 = immutable_list_0.append(var_0)


def test_case_7():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.__add__(immutable_list_0)
    var_0 = immutable_list_1.to_list()
    var_1 = immutable_list_1.find(var_0)
    immutable_list_0.filter(immutable_list_0)


def test_case_8():
    immutable_list_0 = module_0.ImmutableList()
    none_type_0 = None
    bool_0 = immutable_list_0.__eq__(none_type_0)
    immutable_list_0.map(immutable_list_0)


def test_case_9():
    int_0 = 0
    set_0 = {int_0}
    int_1 = 1987
    immutable_list_0 = module_0.ImmutableList(tail=int_1)
    immutable_list_0.map(set_0)


def test_case_10():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_0.filter(immutable_list_0)


def test_case_11():
    int_0 = -2401
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.find(int_0)
    immutable_list_1 = module_0.ImmutableList(is_empty=var_0)
    immutable_list_2 = immutable_list_1.append(var_0)
    immutable_list_2.filter(var_0)


def test_case_12():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.append(immutable_list_0)
    var_0 = immutable_list_1.find(immutable_list_0)
    immutable_list_0.filter(var_0)


def test_case_13():
    bool_0 = True
    immutable_list_0 = module_0.ImmutableList(bool_0, is_empty=bool_0)
    immutable_list_0.find(bool_0)


def test_case_14():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.append(immutable_list_0)
    var_0 = immutable_list_0.find(immutable_list_1)
    var_1 = immutable_list_1.reduce(immutable_list_0, immutable_list_0)
    var_2 = var_1.find(var_0)
    immutable_list_0.filter(immutable_list_0)


def test_case_15():
    immutable_list_0 = module_0.ImmutableList()


def test_case_16():
    none_type_0 = None
    bool_0 = True
    immutable_list_0 = module_0.ImmutableList(is_empty=bool_0)
    immutable_list_1 = immutable_list_0.unshift(none_type_0)


def test_case_17():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    var_0 = immutable_list_1.__add__(immutable_list_1)
    var_1 = immutable_list_1.__len__()
    immutable_list_1.find(var_1)


def test_case_18():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    immutable_list_1.find(immutable_list_0)


def test_case_19():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.append(immutable_list_0)
    immutable_list_2 = immutable_list_0.__add__(immutable_list_0)
    var_0 = immutable_list_2.to_list()
    var_1 = immutable_list_2.find(immutable_list_1)
    var_2 = immutable_list_1.reduce(immutable_list_0, immutable_list_0)
    bool_0 = immutable_list_1.__eq__(immutable_list_2)
    var_1.append(immutable_list_1)


def test_case_20():
    list_0 = []
    none_type_0 = None
    int_0 = 1
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(int_0)
    immutable_list_1.reduce(list_0, none_type_0)


def test_case_21():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    var_0 = immutable_list_0.find(immutable_list_0)
    var_1 = immutable_list_1.__len__()
    immutable_list_1.find(var_1)


def test_case_22():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    bool_0 = immutable_list_0.__eq__(immutable_list_1)
    immutable_list_1.find(immutable_list_0)


def test_case_23():
    bool_0 = False
    immutable_list_0 = module_0.ImmutableList(bool_0, is_empty=bool_0)
    immutable_list_0.reduce(immutable_list_0, immutable_list_0)
