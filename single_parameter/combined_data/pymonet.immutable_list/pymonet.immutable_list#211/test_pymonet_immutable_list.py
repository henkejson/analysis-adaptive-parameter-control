# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.immutable_list as module_0


def test_case_0():
    complex_0 = 1357.5725570026325 + 4122.522077j
    immutable_list_0 = module_0.ImmutableList(complex_0)
    bool_0 = immutable_list_0.__eq__(immutable_list_0)
    immutable_list_0.find(immutable_list_0)


def test_case_1():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = module_0.ImmutableList()
    str_0 = immutable_list_0.__str__()
    bool_0 = immutable_list_1.__eq__(str_0)
    var_0 = immutable_list_0.find(str_0)
    immutable_list_0.filter(var_0)


def test_case_2():
    complex_0 = 1356.472091 + 4122.522077j
    immutable_list_0 = module_0.ImmutableList(complex_0)
    immutable_list_1 = immutable_list_0.unshift(complex_0)


def test_case_3():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = module_0.ImmutableList()
    var_0 = immutable_list_1.__len__()
    immutable_list_0.filter(var_0)


def test_case_4():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    var_0 = immutable_list_1.__len__()
    immutable_list_1.find(var_0)


def test_case_5():
    immutable_list_0 = module_0.ImmutableList()
    str_0 = immutable_list_0.__str__()


def test_case_6():
    complex_0 = 1356.472091 + 4122.522077j
    immutable_list_0 = module_0.ImmutableList(complex_0)
    immutable_list_1 = immutable_list_0.unshift(complex_0)
    var_0 = immutable_list_1.to_list()
    immutable_list_1.find(var_0)


def test_case_7():
    str_0 = "\n        Call success_callback function with monad value when monad is successfully.\n\n        :params success_callback: function to apply with monad value.\n        :type success_callback: Function(A)\n        :returns: self\n        :rtype: Try[A]\n        "
    immutable_list_0 = module_0.ImmutableList(str_0)
    immutable_list_0.map(str_0)


def test_case_8():
    dict_0 = {}
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(dict_0)
    immutable_list_1.map(dict_0)


def test_case_9():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_0.filter(immutable_list_0)


def test_case_10():
    immutable_list_0 = module_0.ImmutableList()
    none_type_0 = None
    var_0 = immutable_list_0.find(none_type_0)
    immutable_list_1 = immutable_list_0.append(immutable_list_0)
    var_1 = immutable_list_0.to_list()
    immutable_list_1.filter(var_1)


def test_case_11():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.find(immutable_list_0)


def test_case_12():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = module_0.ImmutableList()
    none_type_0 = None
    var_0 = immutable_list_0.reduce(none_type_0, immutable_list_1)
    var_1 = immutable_list_0.find(var_0)
    immutable_list_1.filter(var_1)


def test_case_13():
    none_type_0 = None
    bool_0 = True
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.__len__()
    immutable_list_1 = module_0.ImmutableList(bool_0)
    bool_1 = immutable_list_1.__eq__(immutable_list_0)
    immutable_list_1.reduce(none_type_0, bool_0)


def test_case_14():
    immutable_list_0 = module_0.ImmutableList()


def test_case_15():
    str_0 = "\r\t>S9'eQtvqkxj\x0c@Jio"
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.append(str_0)


def test_case_16():
    immutable_list_0 = module_0.ImmutableList()
    str_0 = immutable_list_0.__str__()
    none_type_0 = None
    var_0 = immutable_list_0.reduce(none_type_0, immutable_list_0)
    immutable_list_1 = var_0.unshift(str_0)
    bool_0 = var_0.__eq__(immutable_list_1)
    immutable_list_1.find(var_0)


def test_case_17():
    immutable_list_0 = module_0.ImmutableList()
    list_0 = [immutable_list_0]
    immutable_list_1 = immutable_list_0.unshift(list_0)
    str_0 = immutable_list_0.__str__()
    var_0 = immutable_list_1.append(list_0)
    immutable_list_1.find(immutable_list_1)


def test_case_18():
    complex_0 = 1356.1685768772286 + 4122.522077j
    immutable_list_0 = module_0.ImmutableList(complex_0)
    immutable_list_1 = immutable_list_0.unshift(complex_0)
    var_0 = immutable_list_1.__len__()
    immutable_list_1.find(var_0)


def test_case_19():
    bool_0 = False
    list_0 = [bool_0, bool_0]
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.__len__()
    immutable_list_1 = immutable_list_0.unshift(bool_0)
    immutable_list_1.reduce(list_0, bool_0)


def test_case_20():
    immutable_list_0 = module_0.ImmutableList()
    str_0 = immutable_list_0.__str__()
    none_type_0 = None
    immutable_list_1 = immutable_list_0.unshift(none_type_0)
    bool_0 = immutable_list_1.__eq__(immutable_list_0)
    var_0 = immutable_list_0.reduce(none_type_0, immutable_list_0)
    var_1 = immutable_list_0.find(immutable_list_0)
    immutable_list_0.filter(var_1)


def test_case_21():
    int_0 = -246
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(is_empty=int_0)
    immutable_list_1 = immutable_list_0.append(int_0)
    immutable_list_1.__add__(none_type_0)


def test_case_22():
    complex_0 = 1356.472091 + 4122.522077j
    immutable_list_0 = module_0.ImmutableList(complex_0)
    immutable_list_0.find(immutable_list_0)


def test_case_23():
    immutable_list_0 = module_0.ImmutableList()
    list_0 = [immutable_list_0]
    immutable_list_1 = immutable_list_0.unshift(list_0)
    immutable_list_1.find(immutable_list_1)
