# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.immutable_list as module_0
import builtins as module_1


def test_case_0():
    immutable_list_0 = module_0.ImmutableList()
    bool_0 = immutable_list_0.__eq__(immutable_list_0)
    immutable_list_0.filter(immutable_list_0)


def test_case_1():
    int_0 = -1485
    immutable_list_0 = module_0.ImmutableList(int_0)
    none_type_0 = None
    bool_0 = immutable_list_0.__eq__(int_0)
    immutable_list_0.find(none_type_0)


def test_case_2():
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(none_type_0)
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    immutable_list_1.find(immutable_list_1)


def test_case_3():
    none_type_0 = None
    none_type_1 = None
    immutable_list_0 = module_0.ImmutableList(none_type_1, none_type_1)
    var_0 = immutable_list_0.find(none_type_1)
    immutable_list_0.__add__(none_type_0)


def test_case_4():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.__len__()


def test_case_5():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    var_0 = immutable_list_1.__len__()
    immutable_list_1.find(var_0)


def test_case_6():
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(none_type_0)
    var_0 = immutable_list_0.to_list()


def test_case_7():
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.append(none_type_0)
    var_0 = immutable_list_1.to_list()
    var_0.to_list()


def test_case_8():
    object_0 = module_1.object()
    immutable_list_0 = module_0.ImmutableList()
    bool_0 = immutable_list_0.__eq__(immutable_list_0)
    var_0 = immutable_list_0.__len__()
    var_1 = immutable_list_0.to_list()
    immutable_list_0.map(var_1)


def test_case_9():
    bool_0 = False
    immutable_list_0 = module_0.ImmutableList(tail=bool_0, is_empty=bool_0)
    var_0 = immutable_list_0.__len__()
    immutable_list_1 = module_0.ImmutableList(is_empty=bool_0)
    var_1 = immutable_list_1.__len__()
    var_2 = immutable_list_1.to_list()
    immutable_list_0.map(var_1)


def test_case_10():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_0.filter(immutable_list_0)


def test_case_11():
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(none_type_0, none_type_0)
    immutable_list_1 = immutable_list_0.unshift(none_type_0)
    var_0 = immutable_list_1.find(immutable_list_0)
    tuple_0 = (var_0,)
    immutable_list_2 = module_0.ImmutableList(is_empty=tuple_0)
    immutable_list_3 = module_0.ImmutableList(
        tail=immutable_list_1, is_empty=none_type_0
    )
    immutable_list_1.filter(var_0)


def test_case_12():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.find(immutable_list_0)
    immutable_list_0.filter(immutable_list_0)


def test_case_13():
    int_0 = -1513
    immutable_list_0 = module_0.ImmutableList(int_0)
    none_type_0 = None
    immutable_list_0.find(none_type_0)


def test_case_14():
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(none_type_0)
    var_0 = immutable_list_0.reduce(immutable_list_0, immutable_list_0)
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    immutable_list_1.find(none_type_0)


def test_case_15():
    object_0 = module_1.object()
    set_0 = {object_0, object_0, object_0, object_0}
    bool_0 = False
    immutable_list_0 = module_0.ImmutableList(set_0, is_empty=bool_0)
    immutable_list_0.reduce(object_0, object_0)


def test_case_16():
    immutable_list_0 = module_0.ImmutableList()


def test_case_17():
    bool_0 = True
    immutable_list_0 = module_0.ImmutableList(is_empty=bool_0)
    str_0 = immutable_list_0.__str__()


def test_case_18():
    int_0 = -1499
    immutable_list_0 = module_0.ImmutableList(int_0)
    none_type_0 = None
    bool_0 = immutable_list_0.__len__()
    immutable_list_0.find(none_type_0)


def test_case_19():
    str_0 = "Ouadr?K;1]^XND"
    int_0 = -1915
    immutable_list_0 = module_0.ImmutableList(str_0, int_0)
    immutable_list_0.reduce(str_0, str_0)


def test_case_20():
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    bool_0 = immutable_list_1.__eq__(immutable_list_0)
    immutable_list_1.find(none_type_0)


def test_case_21():
    object_0 = module_1.object()
    immutable_list_0 = module_0.ImmutableList(tail=object_0)
    immutable_list_0.append(immutable_list_0)


def test_case_22():
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(none_type_0)
    var_0 = immutable_list_0.reduce(none_type_0, immutable_list_0)
    immutable_list_1 = var_0.unshift(none_type_0)
    bool_0 = immutable_list_1.__eq__(var_0)
    immutable_list_2 = var_0.unshift(var_0)
    immutable_list_2.find(immutable_list_0)
