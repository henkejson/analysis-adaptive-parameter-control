# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.immutable_list as module_0
import builtins as module_1


def test_case_0():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.append(immutable_list_0)
    bool_0 = immutable_list_0.__eq__(immutable_list_0)
    var_0 = immutable_list_0.to_list()
    immutable_list_0.filter(immutable_list_0)


def test_case_1():
    int_0 = -584
    immutable_list_0 = module_0.ImmutableList(int_0)
    bool_0 = immutable_list_0.__eq__(int_0)
    immutable_list_0.find(immutable_list_0)


def test_case_2():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    immutable_list_1.find(immutable_list_1)


def test_case_3():
    str_0 = '#\n=z~kR-;`!DA."e62.'
    int_0 = 414
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(none_type_0)
    immutable_list_2 = immutable_list_1.unshift(int_0)
    immutable_list_2.__add__(str_0)


def test_case_4():
    bool_0 = False
    immutable_list_0 = module_0.ImmutableList(tail=bool_0)
    var_0 = immutable_list_0.__len__()
    var_1 = immutable_list_0.reduce(var_0, bool_0)


def test_case_5():
    int_0 = 4354
    immutable_list_0 = module_0.ImmutableList(int_0)
    var_0 = immutable_list_0.__len__()
    immutable_list_0.find(var_0)


def test_case_6():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.to_list()
    immutable_list_0.filter(immutable_list_0)


def test_case_7():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    var_0 = immutable_list_1.to_list()
    immutable_list_2 = immutable_list_1.__add__(immutable_list_1)
    immutable_list_2.find(immutable_list_2)


def test_case_8():
    none_type_0 = None
    int_0 = -4
    immutable_list_0 = module_0.ImmutableList(int_0)
    bool_0 = immutable_list_0.__eq__(int_0)
    immutable_list_0.map(none_type_0)


def test_case_9():
    object_0 = module_1.object()
    none_type_0 = None
    bool_0 = True
    immutable_list_0 = module_0.ImmutableList(is_empty=bool_0)
    immutable_list_1 = immutable_list_0.append(object_0)
    immutable_list_1.map(none_type_0)


def test_case_10():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_0.filter(immutable_list_0)


def test_case_11():
    int_0 = -2319
    set_0 = {int_0, int_0, int_0}
    immutable_list_0 = module_0.ImmutableList(tail=set_0)
    var_0 = immutable_list_0.find(immutable_list_0)
    immutable_list_0.filter(var_0)


def test_case_12():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.__len__()
    var_1 = immutable_list_0.find(var_0)
    immutable_list_0.filter(immutable_list_0)


def test_case_13():
    int_0 = -584
    immutable_list_0 = module_0.ImmutableList(int_0)
    immutable_list_0.find(immutable_list_0)


def test_case_14():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.reduce(immutable_list_0, immutable_list_0)
    immutable_list_0.filter(var_0)


def test_case_15():
    none_type_0 = None
    bytes_0 = b""
    bytes_1 = b"\rn6c\xdb\x16h\x01\xe0\xb6\xec2\xf0"
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(bytes_1)
    immutable_list_2 = immutable_list_1.append(bytes_0)
    immutable_list_2.reduce(none_type_0, none_type_0)


def test_case_16():
    immutable_list_0 = module_0.ImmutableList()


def test_case_17():
    immutable_list_0 = module_0.ImmutableList()
    str_0 = immutable_list_0.__str__()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    immutable_list_2 = immutable_list_1.__add__(immutable_list_1)
    immutable_list_2.find(immutable_list_2)


def test_case_18():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    immutable_list_2 = immutable_list_1.__add__(immutable_list_1)
    immutable_list_2.find(immutable_list_2)


def test_case_19():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.to_list()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    var_1 = immutable_list_1.to_list()
    immutable_list_2 = immutable_list_1.__add__(immutable_list_1)
    var_2 = immutable_list_1.__len__()
    immutable_list_2.find(immutable_list_2)


def test_case_20():
    none_type_0 = None
    tuple_0 = ()
    immutable_list_0 = module_0.ImmutableList(tuple_0)
    immutable_list_0.reduce(none_type_0, immutable_list_0)


def test_case_21():
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(is_empty=none_type_0)
    immutable_list_1 = immutable_list_0.append(none_type_0)
    var_0 = immutable_list_0.find(none_type_0)
    var_1 = immutable_list_1.reduce(immutable_list_0, immutable_list_0)
    var_2 = immutable_list_1.find(var_1)
    str_0 = immutable_list_1.__str__()
    var_3 = immutable_list_0.__len__()
    bool_0 = immutable_list_1.__eq__(var_1)
    var_1.filter(immutable_list_0)


def test_case_22():
    none_type_0 = None
    none_type_1 = None
    immutable_list_0 = module_0.ImmutableList(is_empty=none_type_1)
    immutable_list_1 = immutable_list_0.unshift(none_type_0)
    str_0 = immutable_list_1.__str__()
    int_0 = 1
    dict_0 = {int_0: int_0}
    none_type_2 = None
    immutable_list_2 = module_0.ImmutableList()
    immutable_list_3 = immutable_list_2.append(none_type_2)
    str_1 = immutable_list_2.__str__()
    var_0 = immutable_list_3.__len__()
    var_1 = immutable_list_2.to_list()
    var_2 = immutable_list_1.reduce(var_0, var_0)
    immutable_list_4 = immutable_list_0.unshift(int_0)
    var_3 = immutable_list_1.find(var_2)
    bool_0 = immutable_list_3.__eq__(immutable_list_4)
    immutable_list_5 = immutable_list_2.append(none_type_2)
    immutable_list_6 = immutable_list_5.unshift(dict_0)
