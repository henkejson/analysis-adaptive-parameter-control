# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.immutable_list as module_0


def test_case_0():
    bytes_0 = b"\x1b~B\xb0\xde"
    immutable_list_0 = module_0.ImmutableList(bytes_0)
    bool_0 = immutable_list_0.__eq__(immutable_list_0)
    immutable_list_0.find(immutable_list_0)


def test_case_1():
    bytes_0 = b"H\x9bA\xc5\xdd\xc3\xab\x07t"
    immutable_list_0 = module_0.ImmutableList(bytes_0)
    bool_0 = immutable_list_0.__eq__(bytes_0)
    immutable_list_0.find(bytes_0)


def test_case_2():
    bytes_0 = b"\x1b~B\xb0\xde"
    immutable_list_0 = module_0.ImmutableList(bytes_0)
    immutable_list_1 = immutable_list_0.__add__(immutable_list_0)
    immutable_list_1.find(immutable_list_0)


def test_case_3():
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(none_type_0, none_type_0, none_type_0)
    immutable_list_1 = module_0.ImmutableList(tail=immutable_list_0)
    bool_0 = False
    immutable_list_2 = module_0.ImmutableList(is_empty=bool_0)
    var_0 = immutable_list_2.reduce(immutable_list_1, none_type_0)
    immutable_list_3 = module_0.ImmutableList()
    immutable_list_3.__add__(var_0)


def test_case_4():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.__len__()
    var_1 = immutable_list_0.find(immutable_list_0)
    immutable_list_0.filter(var_1)


def test_case_5():
    bytes_0 = b"\x1b~B\xb0\xde"
    immutable_list_0 = module_0.ImmutableList(bytes_0)
    var_0 = immutable_list_0.__len__()
    immutable_list_0.find(var_0)


def test_case_6():
    bool_0 = True
    immutable_list_0 = module_0.ImmutableList(is_empty=bool_0)
    var_0 = immutable_list_0.to_list()


def test_case_7():
    bytes_0 = b"\x1b~B\xb0\xde"
    immutable_list_0 = module_0.ImmutableList(bytes_0)
    immutable_list_1 = immutable_list_0.__add__(immutable_list_0)
    var_0 = immutable_list_1.to_list()
    immutable_list_0.find(immutable_list_0)


def test_case_8():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.to_list()
    str_0 = immutable_list_0.__str__()
    immutable_list_0.map(var_0)


def test_case_9():
    bool_0 = True
    immutable_list_0 = module_0.ImmutableList(is_empty=bool_0)
    bytes_0 = b"\xf7\xdd\xbc{E7\x1d\x1e"
    none_type_0 = None
    var_0 = immutable_list_0.to_list()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    complex_0 = -751.09 + 5840.54j
    tuple_0 = (complex_0, immutable_list_1, immutable_list_0, bytes_0)
    immutable_list_2 = immutable_list_1.append(tuple_0)
    immutable_list_3 = module_0.ImmutableList(bytes_0, is_empty=immutable_list_0)
    bool_1 = immutable_list_2.__eq__(none_type_0)
    immutable_list_4 = immutable_list_1.append(none_type_0)
    str_0 = immutable_list_0.__str__()
    immutable_list_5 = module_0.ImmutableList(is_empty=bytes_0)
    bool_2 = immutable_list_5.__eq__(immutable_list_5)
    var_1 = immutable_list_5.find(bool_0)
    immutable_list_4.map(var_1)


def test_case_10():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_0.filter(immutable_list_0)


def test_case_11():
    immutable_list_0 = module_0.ImmutableList()
    list_0 = [immutable_list_0, immutable_list_0, immutable_list_0]
    str_0 = immutable_list_0.__str__()
    var_0 = immutable_list_0.reduce(list_0, immutable_list_0)
    var_1 = immutable_list_0.find(list_0)
    immutable_list_1 = module_0.ImmutableList(list_0, list_0)
    immutable_list_2 = immutable_list_0.__add__(immutable_list_0)
    immutable_list_2.filter(str_0)


def test_case_12():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.find(immutable_list_0)


def test_case_13():
    bytes_0 = b"\x1b~B\xb0\xde"
    immutable_list_0 = module_0.ImmutableList(bytes_0)
    immutable_list_0.find(immutable_list_0)


def test_case_14():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.find(immutable_list_0)
    var_1 = immutable_list_0.reduce(var_0, immutable_list_0)
    immutable_list_0.filter(var_1)


def test_case_15():
    bool_0 = True
    bytes_0 = b"\x1b~B\xb0\xde"
    immutable_list_0 = module_0.ImmutableList(bytes_0)
    bool_1 = immutable_list_0.__eq__(bool_0)
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    var_0 = immutable_list_0.__len__()
    none_type_0 = None
    immutable_list_2 = module_0.ImmutableList(none_type_0, none_type_0)
    immutable_list_1.reduce(var_0, bool_1)


def test_case_16():
    immutable_list_0 = module_0.ImmutableList()


def test_case_17():
    bytes_0 = b"\x1b~B\xb0\xde"
    immutable_list_0 = module_0.ImmutableList(bytes_0)
    str_0 = immutable_list_0.__str__()
    immutable_list_0.find(immutable_list_0)


def test_case_18():
    bytes_0 = b"\x1b~B\xb0\xde"
    immutable_list_0 = module_0.ImmutableList(bytes_0)
    immutable_list_1 = immutable_list_0.append(bytes_0)
    immutable_list_1.find(immutable_list_1)


def test_case_19():
    bytes_0 = b"H\x9bA\xc5\xdd\x07t"
    immutable_list_0 = module_0.ImmutableList(bytes_0)
    immutable_list_1 = immutable_list_0.unshift(bytes_0)
    immutable_list_0.find(bytes_0)


def test_case_20():
    bytes_0 = b"\x1b~B\xb0\xde"
    immutable_list_0 = module_0.ImmutableList(bytes_0)
    immutable_list_1 = immutable_list_0.append(immutable_list_0)
    immutable_list_2 = immutable_list_1.append(bytes_0)
    immutable_list_1.find(immutable_list_2)


def test_case_21():
    bytes_0 = b"\x1b~B\xb0\xde"
    immutable_list_0 = module_0.ImmutableList(bytes_0)
    immutable_list_1 = immutable_list_0.__add__(immutable_list_0)
    var_0 = immutable_list_1.__len__()
    immutable_list_1.find(immutable_list_0)


def test_case_22():
    bytes_0 = b"\x1b~B\xb0\xde"
    immutable_list_0 = module_0.ImmutableList(bytes_0)
    immutable_list_1 = immutable_list_0.append(immutable_list_0)
    immutable_list_2 = immutable_list_1.append(bytes_0)
    immutable_list_3 = immutable_list_2.unshift(bytes_0)
    bool_0 = immutable_list_2.__eq__(immutable_list_3)
    immutable_list_1.find(immutable_list_2)


def test_case_23():
    bool_0 = True
    immutable_list_0 = module_0.ImmutableList(bool_0)
    immutable_list_0.reduce(bool_0, immutable_list_0)


def test_case_24():
    bytes_0 = b"\x1b~B\xb0\xde"
    immutable_list_0 = module_0.ImmutableList(bytes_0)
    immutable_list_1 = immutable_list_0.__add__(immutable_list_0)
    bool_0 = immutable_list_0.__eq__(immutable_list_1)
    immutable_list_1.find(immutable_list_0)
