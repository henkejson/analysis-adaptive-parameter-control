# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.immutable_list as module_0


def test_case_0():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    bool_0 = immutable_list_0.__eq__(immutable_list_1)
    immutable_list_1.find(immutable_list_1)


def test_case_1():
    immutable_list_0 = module_0.ImmutableList()
    none_type_0 = None
    immutable_list_1 = immutable_list_0.unshift(none_type_0)


def test_case_2():
    none_type_0 = None
    dict_0 = {none_type_0: none_type_0}
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = module_0.ImmutableList(tail=none_type_0)
    immutable_list_1.__add__(dict_0)


def test_case_3():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    var_0 = immutable_list_0.__len__()
    immutable_list_1.find(immutable_list_0)


def test_case_4():
    float_0 = -1682.5553
    immutable_list_0 = module_0.ImmutableList(float_0, float_0)
    immutable_list_0.__len__()


def test_case_5():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    var_0 = immutable_list_0.to_list()
    immutable_list_1.find(immutable_list_0)


def test_case_6():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    var_0 = immutable_list_1.to_list()
    immutable_list_1.find(immutable_list_1)


def test_case_7():
    bool_0 = True
    immutable_list_0 = module_0.ImmutableList(is_empty=bool_0)
    var_0 = immutable_list_0.to_list()
    var_1 = immutable_list_0.reduce(var_0, var_0)
    none_type_0 = None
    var_2 = immutable_list_0.reduce(var_1, none_type_0)
    immutable_list_0.map(bool_0)


def test_case_8():
    bool_0 = True
    immutable_list_0 = module_0.ImmutableList(tail=bool_0)
    none_type_0 = None
    immutable_list_0.map(none_type_0)


def test_case_9():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_0.filter(immutable_list_0)


def test_case_10():
    float_0 = -266.0797
    bytes_0 = b"qg\xc2\xe0\xb3\xf9\t!s\xcf\x1aN\xd8\x12\xab"
    immutable_list_0 = module_0.ImmutableList(tail=bytes_0)
    immutable_list_0.filter(float_0)


def test_case_11():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.find(immutable_list_0)
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    immutable_list_1.find(immutable_list_0)


def test_case_12():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    var_0 = immutable_list_0.reduce(immutable_list_0, immutable_list_0)


def test_case_13():
    none_type_0 = None
    int_0 = -1765
    dict_0 = {int_0: int_0, int_0: int_0}
    int_1 = 2723
    float_0 = 358.0
    immutable_list_0 = module_0.ImmutableList(float_0)
    immutable_list_1 = immutable_list_0.unshift(int_1)
    immutable_list_2 = immutable_list_1.append(dict_0)
    immutable_list_2.reduce(none_type_0, none_type_0)


def test_case_14():
    immutable_list_0 = module_0.ImmutableList()


def test_case_15():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    str_0 = immutable_list_0.__str__()
    immutable_list_1.find(immutable_list_0)


def test_case_16():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.append(immutable_list_0)
    immutable_list_2 = immutable_list_1.unshift(immutable_list_1)
    immutable_list_2.find(immutable_list_0)


def test_case_17():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    bool_0 = immutable_list_0.__eq__(immutable_list_0)
    var_0 = immutable_list_0.__len__()
    immutable_list_1.find(var_0)


def test_case_18():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    immutable_list_2 = immutable_list_1.__add__(immutable_list_0)
    immutable_list_2.find(immutable_list_0)


def test_case_19():
    set_0 = set()
    none_type_0 = None
    bool_0 = True
    immutable_list_0 = module_0.ImmutableList(set_0, is_empty=bool_0)
    immutable_list_0.reduce(none_type_0, none_type_0)


def test_case_20():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    immutable_list_2 = immutable_list_0.__add__(immutable_list_0)
    bool_0 = immutable_list_0.__eq__(immutable_list_2)
    immutable_list_1.find(immutable_list_0)


def test_case_21():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    immutable_list_1.find(immutable_list_0)


def test_case_22():
    int_0 = -3369
    immutable_list_0 = module_0.ImmutableList(int_0, is_empty=int_0)
    var_0 = immutable_list_0.__len__()
    immutable_list_0.find(var_0)


def test_case_23():
    dict_0 = {}
    int_0 = 1
    immutable_list_0 = module_0.ImmutableList(int_0)
    immutable_list_0.find(dict_0)
