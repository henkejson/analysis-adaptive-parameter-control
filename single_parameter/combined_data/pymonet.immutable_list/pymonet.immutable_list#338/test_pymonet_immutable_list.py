# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.immutable_list as module_0
import builtins as module_1


def test_case_0():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    bool_0 = immutable_list_0.__eq__(immutable_list_1)
    immutable_list_1.find(immutable_list_0)


def test_case_1():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    immutable_list_1.find(immutable_list_0)


def test_case_2():
    bytes_0 = b"\xbc\xea0*\x9b\xed\x18uK\x1e\x9c\x0e\xcc\xd0"
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_0.__add__(bytes_0)


def test_case_3():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    immutable_list_2 = immutable_list_0.__len__()
    immutable_list_1.find(immutable_list_2)


def test_case_4():
    dict_0 = {}
    immutable_list_0 = module_0.ImmutableList(dict_0, dict_0, dict_0)
    var_0 = immutable_list_0.__len__()
    immutable_list_0.find(var_0)


def test_case_5():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.to_list()


def test_case_6():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    var_0 = immutable_list_1.to_list()
    immutable_list_1.find(immutable_list_0)


def test_case_7():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.reduce(immutable_list_0, immutable_list_0)
    var_1 = immutable_list_0.to_list()
    immutable_list_0.map(var_1)


def test_case_8():
    none_type_0 = None
    int_0 = 36
    none_type_1 = None
    immutable_list_0 = module_0.ImmutableList(none_type_1)
    immutable_list_1 = immutable_list_0.append(int_0)
    immutable_list_1.map(none_type_0)


def test_case_9():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_0.filter(immutable_list_0)


def test_case_10():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.unshift(immutable_list_0)
    var_0.filter(var_0)


def test_case_11():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    var_0 = immutable_list_0.find(immutable_list_1)
    immutable_list_1.find(immutable_list_0)


def test_case_12():
    object_0 = module_1.object()
    immutable_list_0 = module_0.ImmutableList(object_0, is_empty=object_0)
    immutable_list_1 = immutable_list_0.__add__(immutable_list_0)
    none_type_0 = None
    immutable_list_0.reduce(immutable_list_1, none_type_0)


def test_case_13():
    immutable_list_0 = module_0.ImmutableList()


def test_case_14():
    immutable_list_0 = module_0.ImmutableList()
    str_0 = immutable_list_0.__str__()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    immutable_list_1.find(immutable_list_0)


def test_case_15():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    immutable_list_2 = immutable_list_1.append(immutable_list_0)
    immutable_list_1.find(immutable_list_2)


def test_case_16():
    immutable_list_0 = module_0.ImmutableList()
    bool_0 = immutable_list_0.__eq__(immutable_list_0)
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    immutable_list_1.find(immutable_list_0)


def test_case_17():
    dict_0 = {}
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(dict_0, none_type_0, dict_0)
    var_0 = immutable_list_0.__len__()
    immutable_list_0.find(var_0)


def test_case_18():
    bool_0 = False
    immutable_list_0 = module_0.ImmutableList(bool_0)
    immutable_list_0.find(bool_0)


def test_case_19():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    immutable_list_2 = immutable_list_1.__add__(immutable_list_1)
    bool_0 = immutable_list_1.__eq__(immutable_list_2)
    immutable_list_2.find(immutable_list_2)


def test_case_20():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = module_0.ImmutableList(tail=immutable_list_0)
    var_0 = immutable_list_0.to_list()
    immutable_list_2 = immutable_list_0.unshift(var_0)
    var_1 = immutable_list_0.reduce(var_0, immutable_list_2)
    var_2 = immutable_list_0.__len__()
    var_1.reduce(var_0, var_0)
