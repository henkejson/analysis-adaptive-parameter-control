# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.immutable_list as module_0
import builtins as module_1


def test_case_0():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    var_0 = immutable_list_1.to_list()
    immutable_list_2 = immutable_list_1.unshift(var_0)
    bool_0 = immutable_list_0.__eq__(immutable_list_1)
    immutable_list_1.find(immutable_list_0)


def test_case_1():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    immutable_list_1.find(immutable_list_0)


def test_case_2():
    bytes_0 = b"ZS}\xf0+\x1d5\x04\xd8d$"
    int_0 = 1637
    immutable_list_0 = module_0.ImmutableList(tail=int_0)
    immutable_list_1 = module_0.ImmutableList()
    immutable_list_2 = immutable_list_1.unshift(immutable_list_0)
    immutable_list_2.__add__(bytes_0)


def test_case_3():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.__len__()


def test_case_4():
    int_0 = 0
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(int_0, is_empty=none_type_0)
    var_0 = immutable_list_0.__len__()
    immutable_list_0.find(var_0)


def test_case_5():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    var_0 = immutable_list_1.to_list()
    immutable_list_1.find(immutable_list_0)


def test_case_6():
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(none_type_0)
    immutable_list_1 = immutable_list_0.append(none_type_0)
    immutable_list_2 = immutable_list_0.unshift(none_type_0)
    immutable_list_3 = module_0.ImmutableList()
    immutable_list_0.map(immutable_list_0)


def test_case_7():
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(is_empty=none_type_0)
    immutable_list_1 = immutable_list_0.append(none_type_0)
    immutable_list_2 = immutable_list_0.__add__(immutable_list_0)
    immutable_list_3 = module_0.ImmutableList(tail=none_type_0)
    bool_0 = True
    var_0 = immutable_list_1.reduce(bool_0, none_type_0)
    var_1 = immutable_list_3.find(var_0)
    bool_1 = immutable_list_0.__eq__(bool_0)
    immutable_list_4 = immutable_list_3.append(none_type_0)
    immutable_list_4.map(none_type_0)


def test_case_8():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_0.filter(immutable_list_0)


def test_case_9():
    int_0 = 0
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(none_type_0, none_type_0)
    immutable_list_1 = immutable_list_0.append(int_0)
    bool_0 = immutable_list_0.__eq__(immutable_list_0)
    var_0 = immutable_list_0.unshift(int_0)
    int_1 = 57
    var_0.filter(int_1)


def test_case_10():
    bool_0 = False
    immutable_list_0 = module_0.ImmutableList()
    none_type_0 = None
    immutable_list_1 = immutable_list_0.unshift(none_type_0)
    var_0 = immutable_list_0.find(bool_0)


def test_case_11():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.reduce(immutable_list_0, immutable_list_0)
    immutable_list_0.filter(var_0)


def test_case_12():
    bool_0 = True
    none_type_0 = None
    bytes_0 = b"+\xc6\x1c\xef\xe7\x19\xc3\xcb\x014\xf4\xec\xc5\xbe\x05\x08\x9d\x02"
    immutable_list_0 = module_0.ImmutableList(bytes_0, is_empty=bytes_0)
    immutable_list_1 = immutable_list_0.append(none_type_0)
    immutable_list_1.reduce(bool_0, bool_0)


def test_case_13():
    immutable_list_0 = module_0.ImmutableList()


def test_case_14():
    dict_0 = {}
    immutable_list_0 = module_0.ImmutableList(dict_0)
    str_0 = immutable_list_0.__str__()
    immutable_list_0.find(dict_0)


def test_case_15():
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(tail=none_type_0)
    immutable_list_1 = immutable_list_0.append(none_type_0)
    var_0 = immutable_list_0.__len__()


def test_case_16():
    immutable_list_0 = module_0.ImmutableList()
    str_0 = immutable_list_0.__str__()
    immutable_list_1 = immutable_list_0.__add__(immutable_list_0)
    var_0 = immutable_list_1.to_list()
    var_1 = immutable_list_1.reduce(var_0, immutable_list_1)
    immutable_list_2 = immutable_list_1.append(immutable_list_0)
    bool_0 = immutable_list_2.__eq__(immutable_list_1)
    bool_1 = immutable_list_0.__len__()
    immutable_list_3 = immutable_list_0.__add__(immutable_list_0)
    bool_2 = immutable_list_1.__eq__(immutable_list_0)
    bool_3 = immutable_list_0.__eq__(bool_1)
    immutable_list_4 = var_1.unshift(bool_3)


def test_case_17():
    bool_0 = True
    immutable_list_0 = module_0.ImmutableList(bool_0)
    immutable_list_1 = immutable_list_0.append(bool_0)
    var_0 = immutable_list_1.__len__()
    immutable_list_1.find(var_0)


def test_case_18():
    tuple_0 = ()
    object_0 = module_1.object()
    none_type_0 = None
    dict_0 = {object_0: none_type_0, none_type_0: none_type_0}
    list_0 = [dict_0]
    immutable_list_0 = module_0.ImmutableList(object_0)
    var_0 = immutable_list_0.to_list()
    immutable_list_1 = module_0.ImmutableList(list_0)
    immutable_list_0.reduce(list_0, tuple_0)


def test_case_19():
    dict_0 = {}
    immutable_list_0 = module_0.ImmutableList(dict_0)
    immutable_list_0.find(dict_0)
