# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.immutable_list as module_0
import builtins as module_1


def test_case_0():
    immutable_list_0 = module_0.ImmutableList()
    bool_0 = immutable_list_0.__eq__(immutable_list_0)
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    immutable_list_1.find(immutable_list_0)


def test_case_1():
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList()
    bool_0 = immutable_list_0.__eq__(none_type_0)
    immutable_list_1 = module_0.ImmutableList(bool_0)
    immutable_list_1.find(bool_0)


def test_case_2():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.append(immutable_list_0)
    var_0 = immutable_list_1.find(immutable_list_0)
    immutable_list_1.filter(var_0)


def test_case_3():
    list_0 = []
    bool_0 = True
    immutable_list_0 = module_0.ImmutableList(tail=bool_0)
    immutable_list_0.__add__(list_0)


def test_case_4():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.__len__()
    immutable_list_1 = immutable_list_0.unshift(var_0)
    immutable_list_1.find(var_0)


def test_case_5():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.to_list()


def test_case_6():
    complex_0 = -1366 - 890.89174j
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(none_type_0)
    immutable_list_1 = immutable_list_0.unshift(complex_0)
    object_0 = module_1.object()
    none_type_1 = None
    immutable_list_2 = module_0.ImmutableList(none_type_1)
    immutable_list_2.map(object_0)


def test_case_7():
    bytes_0 = b"_\xc0\x1c\x92YK\xf2\x06w\x06aBS\xa10\x9d\xbb\xcb\x9c:"
    dict_0 = {}
    bool_0 = True
    immutable_list_0 = module_0.ImmutableList(tail=dict_0, is_empty=bool_0)
    immutable_list_1 = immutable_list_0.unshift(bytes_0)
    immutable_list_0.map(dict_0)


def test_case_8():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_0.filter(immutable_list_0)


def test_case_9():
    str_0 = 'F"G/M\t,=f0?|'
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(is_empty=none_type_0)
    var_0 = immutable_list_0.find(str_0)


def test_case_10():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.reduce(immutable_list_0, immutable_list_0)
    immutable_list_1 = immutable_list_0.unshift(var_0)
    immutable_list_1.find(var_0)


def test_case_11():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.to_list()
    immutable_list_1 = module_0.ImmutableList()
    immutable_list_2 = immutable_list_0.unshift(var_0)
    immutable_list_2.reduce(var_0, immutable_list_0)


def test_case_12():
    immutable_list_0 = module_0.ImmutableList()


def test_case_13():
    immutable_list_0 = module_0.ImmutableList()
    str_0 = immutable_list_0.__str__()
    bool_0 = immutable_list_0.__eq__(immutable_list_0)
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    immutable_list_1.find(immutable_list_0)


def test_case_14():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    immutable_list_1.find(immutable_list_1)


def test_case_15():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    bool_0 = immutable_list_1.__eq__(immutable_list_0)
    immutable_list_2 = immutable_list_0.__add__(immutable_list_0)
    immutable_list_1.find(immutable_list_1)


def test_case_16():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.append(immutable_list_0)
    var_0 = immutable_list_1.__len__()
    var_1 = immutable_list_1.find(immutable_list_1)
    immutable_list_2 = immutable_list_1.append(immutable_list_1)
    str_0 = immutable_list_0.__str__()
    var_2 = immutable_list_2.__len__()
    immutable_list_1.filter(var_0)


def test_case_17():
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(none_type_0)
    immutable_list_2 = immutable_list_1.unshift(immutable_list_0)
    var_0 = immutable_list_1.find(immutable_list_0)
    bool_0 = immutable_list_0.__eq__(immutable_list_1)
    immutable_list_0.filter(var_0)


def test_case_18():
    bool_0 = True
    immutable_list_0 = module_0.ImmutableList(bool_0)
    var_0 = immutable_list_0.__len__()
    immutable_list_0.find(bool_0)


def test_case_19():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.append(immutable_list_0)
    var_0 = immutable_list_1.find(immutable_list_0)
    var_1 = immutable_list_1.to_list()
    immutable_list_1.filter(var_0)


def test_case_20():
    str_0 = "\n        Call success_callback function with monad value when monad is not successfully.\n\n        :params fail_callback: function to apply with monad value.\n        :type fail_callback: Function(A)\n        :returns: self\n        :rtype: Try[A]\n        "
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(none_type_0)
    immutable_list_1 = immutable_list_0.append(str_0)
    immutable_list_2 = module_0.ImmutableList(str_0, none_type_0, none_type_0)
    immutable_list_2.reduce(none_type_0, immutable_list_2)


def test_case_21():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    bool_0 = immutable_list_1.__len__()
    immutable_list_1.find(immutable_list_1)


def test_case_22():
    bool_0 = True
    immutable_list_0 = module_0.ImmutableList(bool_0)
    immutable_list_0.find(bool_0)
