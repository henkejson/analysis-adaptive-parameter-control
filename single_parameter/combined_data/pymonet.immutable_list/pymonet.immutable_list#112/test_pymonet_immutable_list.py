# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.immutable_list as module_0
import typing as module_1


def test_case_0():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    bool_0 = immutable_list_1.__eq__(immutable_list_1)
    immutable_list_1.find(immutable_list_0)


def test_case_1():
    bytes_0 = b"yk\xe3\xc1`"
    immutable_list_0 = module_0.ImmutableList(bytes_0)
    bool_0 = immutable_list_0.__eq__(bytes_0)
    var_0 = immutable_list_0.__len__()
    immutable_list_0.find(bytes_0)


def test_case_2():
    generic_0 = module_1.Generic()
    set_0 = {generic_0, generic_0, generic_0, generic_0}
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(set_0)


def test_case_3():
    dict_0 = {}
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(tail=dict_0)
    immutable_list_0.__add__(none_type_0)


def test_case_4():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.__len__()
    immutable_list_0.filter(immutable_list_0)


def test_case_5():
    bytes_0 = b"yk\xe3\xc1`"
    immutable_list_0 = module_0.ImmutableList(bytes_0)
    var_0 = immutable_list_0.__len__()
    immutable_list_0.find(bytes_0)


def test_case_6():
    immutable_list_0 = module_0.ImmutableList()
    str_0 = immutable_list_0.to_list()
    immutable_list_0.filter(immutable_list_0)


def test_case_7():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.__add__(immutable_list_0)
    str_0 = immutable_list_1.__str__()
    immutable_list_1.filter(immutable_list_1)


def test_case_8():
    generic_0 = module_1.Generic()
    dict_0 = {generic_0: generic_0, generic_0: generic_0}
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList(none_type_0)
    var_0 = immutable_list_0.to_list()
    immutable_list_0.map(dict_0)


def test_case_9():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.to_list()
    bytes_0 = b"\x7fA"
    immutable_list_1 = module_0.ImmutableList()
    immutable_list_2 = immutable_list_1.unshift(bytes_0)
    var_1 = immutable_list_2.to_list()
    immutable_list_3 = immutable_list_1.append(var_1)
    bool_0 = var_1.__eq__(var_1)
    bool_1 = immutable_list_3.__eq__(immutable_list_2)
    var_2 = immutable_list_1.reduce(var_1, var_1)
    var_3 = immutable_list_1.find(var_2)
    str_0 = var_1.__str__()
    immutable_list_4 = immutable_list_0.unshift(var_0)
    immutable_list_4.map(bool_1)


def test_case_10():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_0.filter(immutable_list_0)


def test_case_11():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.__len__()
    var_1 = immutable_list_0.find(var_0)
    immutable_list_0.filter(var_1)


def test_case_12():
    bytes_0 = b"yk\xe3\xc1`"
    immutable_list_0 = module_0.ImmutableList(bytes_0)
    immutable_list_0.find(bytes_0)


def test_case_13():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.__len__()
    var_1 = immutable_list_0.find(var_0)
    var_2 = immutable_list_0.reduce(var_1, immutable_list_0)
    var_3 = immutable_list_0.__len__()
    immutable_list_0.filter(immutable_list_0)


def test_case_14():
    bytes_0 = b")d"
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList()
    bool_0 = immutable_list_0.__eq__(none_type_0)
    immutable_list_1 = module_0.ImmutableList(bytes_0)
    immutable_list_1.reduce(immutable_list_0, immutable_list_0)


def test_case_15():
    immutable_list_0 = module_0.ImmutableList()


def test_case_16():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.append(immutable_list_0)
    immutable_list_2 = immutable_list_0.unshift(immutable_list_0)
    immutable_list_3 = immutable_list_2.__add__(immutable_list_0)
    immutable_list_3.find(immutable_list_3)


def test_case_17():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    bool_0 = immutable_list_0.__eq__(immutable_list_1)
    immutable_list_1.find(immutable_list_0)


def test_case_18():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    bool_0 = immutable_list_1.__len__()
    immutable_list_1.find(immutable_list_0)


def test_case_19():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.append(immutable_list_0)
    bool_0 = immutable_list_0.__eq__(immutable_list_1)
    immutable_list_2 = immutable_list_0.append(immutable_list_1)
    immutable_list_3 = immutable_list_0.unshift(immutable_list_0)
    immutable_list_4 = immutable_list_3.__add__(immutable_list_0)
    immutable_list_4.find(immutable_list_4)


def test_case_20():
    dict_0 = {}
    immutable_list_0 = module_0.ImmutableList(dict_0)
    tuple_0 = (dict_0,)
    immutable_list_1 = module_0.ImmutableList()
    immutable_list_2 = immutable_list_1.append(tuple_0)
    str_0 = "BK{"
    immutable_list_3 = module_0.ImmutableList(str_0, str_0)
    var_0 = immutable_list_0.to_list()
    immutable_list_3.reduce(var_0, tuple_0)
