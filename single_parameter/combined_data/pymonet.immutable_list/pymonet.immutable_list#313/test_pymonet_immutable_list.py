# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.immutable_list as module_0


def test_case_0():
    bytes_0 = b"\xdd\x1b!\x82\xef\xfc\xaf\xaf:Y\n5\xd4\xe9e\xc3>!\xf3\xbe"
    immutable_list_0 = module_0.ImmutableList(is_empty=bytes_0)
    bool_0 = immutable_list_0.__eq__(immutable_list_0)
    immutable_list_1 = immutable_list_0.unshift(bytes_0)
    immutable_list_1.find(immutable_list_1)


def test_case_1():
    bytes_0 = b"\xdd\x1b\x82\xfc\xaf\xaf:Y\n5\xd4\xe9e\xc3>!\xf3\xbe"
    immutable_list_0 = module_0.ImmutableList(is_empty=bytes_0)
    bool_0 = immutable_list_0.__eq__(bytes_0)
    immutable_list_1 = immutable_list_0.unshift(bytes_0)
    immutable_list_1.find(bytes_0)


def test_case_2():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.__add__(immutable_list_0)


def test_case_3():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.__len__()
    immutable_list_0.__add__(var_0)


def test_case_4():
    immutable_list_0 = module_0.ImmutableList()
    str_0 = immutable_list_0.__len__()


def test_case_5():
    none_type_0 = None
    dict_0 = {}
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(dict_0)
    var_0 = immutable_list_1.__len__()
    immutable_list_1.find(none_type_0)


def test_case_6():
    immutable_list_0 = module_0.ImmutableList()
    str_0 = immutable_list_0.__str__()


def test_case_7():
    bytes_0 = b"\xdd\x1b!\x82\xef\xfc\xaf\xaf:Y\n5\xd4\xe9e\xc3>!\xf3\xbe"
    immutable_list_0 = module_0.ImmutableList(is_empty=bytes_0)
    str_0 = immutable_list_0.__str__()
    immutable_list_1 = immutable_list_0.unshift(bytes_0)
    str_1 = immutable_list_1.__str__()
    immutable_list_1.find(bytes_0)


def test_case_8():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_0.map(immutable_list_0)


def test_case_9():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_0.filter(immutable_list_0)


def test_case_10():
    none_type_0 = None
    none_type_1 = None
    immutable_list_0 = module_0.ImmutableList(tail=none_type_1)
    var_0 = immutable_list_0.find(none_type_0)
    immutable_list_1 = module_0.ImmutableList()
    immutable_list_1.filter(immutable_list_1)


def test_case_11():
    float_0 = -1626.839596
    immutable_list_0 = module_0.ImmutableList(float_0)
    immutable_list_0.find(float_0)


def test_case_12():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.reduce(immutable_list_0, immutable_list_0)
    var_0.filter(immutable_list_0)


def test_case_13():
    immutable_list_0 = module_0.ImmutableList()


def test_case_14():
    int_0 = 510
    bytes_0 = b"\xdd\x1b!\x82\xef\xfc\xaf\xaf:Y\n5\xd4\xe9e\xc3>!\xf3\xbe"
    immutable_list_0 = module_0.ImmutableList(is_empty=bytes_0)
    immutable_list_1 = immutable_list_0.append(int_0)
    immutable_list_2 = immutable_list_1.__add__(immutable_list_1)
    str_0 = immutable_list_0.__str__()
    immutable_list_3 = immutable_list_0.unshift(bytes_0)
    immutable_list_3.find(bytes_0)


def test_case_15():
    bool_0 = False
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(none_type_0)
    immutable_list_2 = immutable_list_1.unshift(bool_0)


def test_case_16():
    bytes_0 = b"\xdd\x1b\x82\xfc\xaf\xaf:Y\n5\xd4\xe9e\xc3>!\xf3\xbe"
    immutable_list_0 = module_0.ImmutableList(is_empty=bytes_0)
    bool_0 = immutable_list_0.__eq__(bytes_0)
    immutable_list_1 = immutable_list_0.unshift(bytes_0)
    immutable_list_2 = immutable_list_1.__add__(immutable_list_1)
    immutable_list_1.find(bytes_0)


def test_case_17():
    bytes_0 = b"\xdd\x1b!\x82\xef\xfc\xaf\xaf:Y\n5\xd4\xe9e\xc3>!\xf3\xbe"
    immutable_list_0 = module_0.ImmutableList(is_empty=bytes_0)
    immutable_list_1 = immutable_list_0.__add__(immutable_list_0)
    bool_0 = immutable_list_1.__eq__(bytes_0)
    bool_1 = immutable_list_0.__eq__(immutable_list_1)
    immutable_list_2 = immutable_list_1.unshift(bytes_0)
    immutable_list_2.find(immutable_list_2)


def test_case_18():
    complex_0 = 3644.106 + 1796j
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.append(complex_0)
    immutable_list_2 = module_0.ImmutableList()
    str_0 = immutable_list_2.__str__()
    immutable_list_3 = immutable_list_2.__add__(immutable_list_2)
    var_0 = immutable_list_3.to_list()
    immutable_list_4 = immutable_list_2.__add__(immutable_list_2)
    none_type_0 = None
    str_1 = immutable_list_2.__str__()
    immutable_list_4.map(none_type_0)


def test_case_19():
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.__len__()
    immutable_list_1 = immutable_list_0.__add__(immutable_list_0)
    var_1 = immutable_list_0.__len__()
    var_2 = immutable_list_1.find(var_0)
    var_3 = immutable_list_0.reduce(var_2, none_type_0)
    immutable_list_1.filter(var_2)


def test_case_20():
    bool_0 = False
    bool_1 = True
    immutable_list_0 = module_0.ImmutableList(bool_1)
    immutable_list_0.reduce(bool_0, bool_0)


def test_case_21():
    none_type_0 = None
    int_0 = 1379
    bytes_0 = b"\xdd\x1b!\x82\xef\xfc\xaf\xaf:Y\n5\xd4\xe9e\xc3>!\xf3\xbe"
    immutable_list_0 = module_0.ImmutableList(is_empty=bytes_0)
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    immutable_list_1.reduce(none_type_0, int_0)


def test_case_22():
    bytes_0 = b"c)\xab\xe1"
    immutable_list_0 = module_0.ImmutableList(bytes_0)
    var_0 = immutable_list_0.__len__()
    immutable_list_0.find(bytes_0)


def test_case_23():
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.__len__()
    immutable_list_1 = immutable_list_0.__add__(immutable_list_0)
    var_1 = immutable_list_1.reduce(var_0, var_0)
    var_2 = immutable_list_1.find(var_0)
    var_3 = immutable_list_0.reduce(var_2, none_type_0)
    immutable_list_2 = immutable_list_0.unshift(var_1)
    bool_0 = immutable_list_1.__eq__(immutable_list_2)
    immutable_list_2.find(immutable_list_2)
