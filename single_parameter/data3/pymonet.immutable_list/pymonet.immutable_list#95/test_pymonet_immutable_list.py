# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.immutable_list as module_0
import builtins as module_1


def test_case_0():
    immutable_list_0 = module_0.ImmutableList()
    none_type_0 = None
    immutable_list_1 = immutable_list_0.append(immutable_list_0)
    immutable_list_2 = immutable_list_0.unshift(none_type_0)
    bool_0 = immutable_list_0.__eq__(immutable_list_0)
    str_0 = immutable_list_0.__str__()
    bool_1 = immutable_list_0.__eq__(immutable_list_0)


def test_case_1():
    float_0 = 2905.97163
    immutable_list_0 = module_0.ImmutableList(float_0, float_0, float_0)
    bool_0 = immutable_list_0.__eq__(float_0)
    immutable_list_0.find(immutable_list_0)


def test_case_2():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.append(immutable_list_0)


def test_case_3():
    bytes_0 = b"\x80\xfe\xc37\xdc\xc4P1/\xa9W"
    int_0 = 2189
    bool_0 = True
    tuple_0 = (bool_0,)
    dict_0 = {tuple_0: tuple_0}
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(dict_0, none_type_0)
    immutable_list_1 = immutable_list_0.append(int_0)
    immutable_list_1.__add__(bytes_0)


def test_case_4():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.__len__()


def test_case_5():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.to_list()
    immutable_list_0.filter(var_0)


def test_case_6():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    var_0 = immutable_list_1.to_list()
    immutable_list_0.map(immutable_list_1)


def test_case_7():
    int_0 = -358
    set_0 = set()
    list_0 = [int_0, set_0, set_0, set_0]
    dict_0 = {}
    immutable_list_0 = module_0.ImmutableList(tail=dict_0)
    var_0 = immutable_list_0.find(list_0)
    immutable_list_0.map(var_0)


def test_case_8():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_0.filter(immutable_list_0)


def test_case_9():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.__add__(immutable_list_0)
    var_0 = module_1.object()
    bool_0 = immutable_list_0.__eq__(var_0)
    immutable_list_2 = module_0.ImmutableList()
    var_1 = immutable_list_2.to_list()
    var_2 = immutable_list_0.find(immutable_list_1)
    immutable_list_1.filter(immutable_list_0)


def test_case_10():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.find(immutable_list_0)


def test_case_11():
    float_0 = 2905.97163
    immutable_list_0 = module_0.ImmutableList(float_0, float_0, float_0)
    immutable_list_0.find(immutable_list_0)


def test_case_12():
    bool_0 = True
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = module_0.ImmutableList(is_empty=bool_0)
    none_type_0 = None
    var_0 = immutable_list_0.reduce(none_type_0, none_type_0)
    immutable_list_2 = immutable_list_1.append(bool_0)
    var_1 = immutable_list_1.to_list()


def test_case_13():
    bool_0 = True
    set_0 = {bool_0}
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(set_0, none_type_0, bool_0)
    var_0 = immutable_list_0.to_list()
    str_0 = immutable_list_0.__str__()
    immutable_list_1 = immutable_list_0.unshift(bool_0)
    none_type_1 = None
    str_1 = var_0.__str__()
    immutable_list_2 = module_0.ImmutableList(set_0, none_type_1)
    immutable_list_2.reduce(immutable_list_2, bool_0)


def test_case_14():
    immutable_list_0 = module_0.ImmutableList()


def test_case_15():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)


def test_case_16():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    var_0 = immutable_list_0.to_list()
    bool_0 = immutable_list_0.__eq__(immutable_list_1)
    immutable_list_2 = module_0.ImmutableList(var_0)
    immutable_list_2.find(var_0)


def test_case_17():
    float_0 = 2905.97163
    immutable_list_0 = module_0.ImmutableList(float_0, float_0, float_0)
    immutable_list_1 = immutable_list_0.append(float_0)
    immutable_list_0.find(immutable_list_0)


def test_case_18():
    bool_0 = False
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    immutable_list_2 = immutable_list_0.unshift(bool_0)
    immutable_list_2.reduce(bool_0, immutable_list_2)


def test_case_19():
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.find(immutable_list_0)
    immutable_list_1 = immutable_list_0.unshift(var_0)
    bool_0 = immutable_list_0.__eq__(immutable_list_1)
    immutable_list_2 = module_0.ImmutableList(none_type_0)
    immutable_list_3 = module_0.ImmutableList()
    var_1 = immutable_list_2.find(immutable_list_3)
    immutable_list_0.filter(immutable_list_1)


def test_case_20():
    float_0 = 269.0
    immutable_list_0 = module_0.ImmutableList(float_0)
    var_0 = immutable_list_0.__len__()
    immutable_list_0.find(var_0)


def test_case_21():
    float_0 = 269.0
    immutable_list_0 = module_0.ImmutableList(float_0)
    immutable_list_0.find(immutable_list_0)


def test_case_22():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = module_0.ImmutableList()
    none_type_0 = None
    immutable_list_2 = immutable_list_1.append(none_type_0)
    immutable_list_3 = immutable_list_0.__add__(immutable_list_1)
    immutable_list_4 = immutable_list_0.unshift(immutable_list_3)
    immutable_list_5 = immutable_list_4.__add__(immutable_list_0)
    str_0 = immutable_list_5.__len__()
    immutable_list_1.map(none_type_0)
