# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.maybe as module_0
import typing as module_1


def test_case_0():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)


def test_case_1():
    bool_0 = False
    maybe_0 = module_0.Maybe(bool_0, bool_0)


def test_case_2():
    bool_0 = True
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.map(none_type_0)
    var_1 = maybe_0.to_box()
    var_2 = var_0.ap(var_1)
    maybe_1 = module_0.Maybe(bool_0, bool_0)
    maybe_2 = module_0.Maybe(none_type_0, maybe_0)
    bool_1 = maybe_2.__eq__(bool_0)
    var_3 = maybe_0.get_or_else(bool_0)
    complex_0 = 1814.697 + 1246.4496j
    bool_2 = maybe_1.__eq__(complex_0)
    var_4 = maybe_0.map(bool_2)
    maybe_3 = module_0.Maybe(none_type_0, bool_2)
    var_5 = maybe_3.to_either()
    bool_3 = True
    var_6 = maybe_0.filter(bool_2)
    var_7 = var_4.filter(var_4)
    maybe_4 = module_0.Maybe(bool_0, bool_3)
    maybe_5 = module_0.Maybe(var_4, bool_2)
    var_8 = maybe_2.ap(var_5)
    var_9 = var_0.to_box()
    bool_4 = var_0.__eq__(var_6)


def test_case_3():
    dict_0 = {}
    bool_0 = True
    maybe_0 = module_0.Maybe(dict_0, bool_0)
    maybe_1 = module_0.Maybe(dict_0, dict_0)
    var_0 = maybe_0.to_validation()
    var_1 = maybe_0.ap(maybe_1)
    var_2 = maybe_0.to_lazy()
    bool_1 = var_2.__eq__(maybe_1)
    list_0 = [maybe_1, maybe_0, maybe_0, dict_0]
    none_type_0 = None
    maybe_2 = module_0.Maybe(none_type_0, var_2)
    var_3 = maybe_0.ap(list_0)
    bool_2 = var_3.__eq__(bool_0)


def test_case_4():
    bool_0 = True
    dict_0 = {bool_0: bool_0}
    maybe_0 = module_0.Maybe(dict_0, dict_0)
    var_0 = maybe_0.get_or_else(maybe_0)
    var_1 = maybe_0.to_lazy()
    var_2 = maybe_0.map(var_1)
    var_3 = var_2.filter(var_0)
    var_4 = var_3.to_try()
    var_5 = maybe_0.map(var_4)
    var_6 = maybe_0.ap(maybe_0)


def test_case_5():
    int_0 = -341
    bool_0 = False
    maybe_0 = module_0.Maybe(int_0, bool_0)
    var_0 = maybe_0.to_validation()
    maybe_0.map(var_0)


def test_case_6():
    int_0 = -3314
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, int_0)
    var_0 = maybe_0.to_validation()
    maybe_1 = module_0.Maybe(int_0, bool_0)
    var_1 = maybe_1.to_try()
    var_2 = maybe_1.bind(int_0)
    var_3 = maybe_1.to_box()
    var_4 = var_2.to_try()
    maybe_2 = module_0.Maybe(var_1, int_0)
    var_4.to_box()


def test_case_7():
    complex_0 = -1222.8867 + 1217.738j
    bool_0 = False
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    maybe_0.bind(complex_0)


def test_case_8():
    none_type_0 = None
    set_0 = set()
    maybe_0 = module_0.Maybe(set_0, none_type_0)
    var_0 = maybe_0.to_either()
    var_1 = var_0.to_lazy()
    var_2 = var_0.to_try()
    maybe_0.ap(maybe_0)


def test_case_9():
    bool_0 = True
    none_type_0 = None
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    bool_1 = maybe_0.__eq__(bool_0)
    maybe_1 = module_0.Maybe(none_type_0, bool_1)
    var_0 = maybe_1.to_either()
    var_1 = maybe_0.filter(bool_1)
    maybe_2 = module_0.Maybe(var_1, bool_1)
    var_2 = var_0.to_box()
    maybe_2.filter(var_1)


def test_case_10():
    list_0 = []
    bool_0 = False
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.get_or_else(list_0)
    maybe_1 = module_0.Maybe(var_0, var_0)
    var_1 = maybe_1.to_box()


def test_case_11():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.get_or_else(maybe_0)
    maybe_1 = module_0.Maybe(none_type_0, maybe_0)
    var_1 = maybe_1.to_either()
    var_0.bind(var_0)


def test_case_12():
    bytes_0 = b"\xfa\xbc\x9d\xecTH\x94/0\xe1\xad00\xdf\xb8\xf3\x08\x07\x83"
    maybe_0 = module_0.Maybe(bytes_0, bytes_0)
    var_0 = maybe_0.to_box()
    var_1 = var_0.to_try()
    var_1.to_box()


def test_case_13():
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.to_lazy()
    var_0.filter(var_0)


def test_case_14():
    bool_0 = False
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.to_lazy()


def test_case_15():
    bytes_0 = b"5\xda\xb8"
    maybe_0 = module_0.Maybe(bytes_0, bytes_0)
    var_0 = maybe_0.to_try()


def test_case_16():
    int_0 = 1317
    bool_0 = False
    maybe_0 = module_0.Maybe(int_0, bool_0)
    var_0 = maybe_0.to_try()
    var_0.to_try()


def test_case_17():
    bool_0 = False
    list_0 = [bool_0]
    maybe_0 = module_0.Maybe(list_0, bool_0)
    var_0 = maybe_0.to_validation()
    var_0.filter(maybe_0)


def test_case_18():
    dict_0 = {}
    bool_0 = True
    maybe_0 = module_0.Maybe(dict_0, bool_0)
    maybe_1 = module_0.Maybe(dict_0, dict_0)
    var_0 = maybe_0.to_validation()
    var_1 = maybe_0.ap(maybe_1)
    var_2 = maybe_0.to_lazy()
    bool_1 = var_2.__eq__(maybe_1)
    list_0 = [maybe_1, maybe_0, maybe_0, dict_0]
    none_type_0 = None
    maybe_2 = module_0.Maybe(none_type_0, var_2)
    var_3 = maybe_0.ap(list_0)
    var_4 = var_2.to_box()
    bool_2 = var_3.__eq__(bool_0)


def test_case_19():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    bool_0 = maybe_0.__eq__(maybe_0)
    bool_1 = False
    var_0 = maybe_0.to_either()
    var_1 = maybe_0.to_either()
    maybe_1 = var_0.to_try()
    bool_2 = maybe_0.__eq__(none_type_0)
    var_2 = maybe_0.get_or_else(bool_1)
    complex_0 = 1814.697 + 1246.4496j
    bool_3 = var_1.__eq__(var_2)
    var_0.get_or_else(complex_0)


def test_case_20():
    bool_0 = False
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.to_lazy()
    var_1 = var_0.to_validation()
    var_1.get_or_else(bool_0)


def test_case_21():
    bool_0 = True
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.map(none_type_0)
    var_1 = var_0.bind(maybe_0)
    var_2 = var_0.ap(var_1)
    maybe_1 = module_0.Maybe(none_type_0, maybe_0)
    bool_1 = maybe_1.__eq__(bool_0)
    bool_2 = maybe_1.__eq__(bool_1)
    var_3 = var_1.get_or_else(none_type_0)
    var_4 = var_0.to_either()
    var_5 = maybe_0.to_lazy()
    var_6 = maybe_0.filter(bool_2)
    maybe_2 = module_0.Maybe(var_6, bool_2)
    generic_0 = module_1.Generic()
    bool_3 = maybe_1.__eq__(maybe_0)
    var_7 = var_2.to_validation()
    var_8 = var_0.to_either()
    maybe_2.filter(generic_0)
