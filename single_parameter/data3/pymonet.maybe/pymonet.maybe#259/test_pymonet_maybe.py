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
    complex_0 = 182.601 + 172.11j
    maybe_0 = module_0.Maybe(complex_0, complex_0)
    bool_0 = maybe_0.__eq__(maybe_0)
    var_0 = maybe_0.filter(bool_0)
    bool_1 = False
    maybe_1 = module_0.Maybe(bool_0, bool_1)
    maybe_1.filter(complex_0)


def test_case_3():
    int_0 = 0
    bool_0 = True
    maybe_0 = module_0.Maybe(int_0, bool_0)
    var_0 = maybe_0.to_either()
    set_0 = {int_0, int_0, int_0, int_0}
    var_1 = var_0.to_lazy()
    none_type_0 = None
    maybe_1 = module_0.Maybe(set_0, none_type_0)
    var_2 = maybe_1.to_lazy()
    var_3 = var_2.to_validation()
    var_4 = var_2.to_validation()
    bool_1 = maybe_1.__eq__(int_0)
    var_5 = maybe_1.to_validation()


def test_case_4():
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    bytes_0 = b"dhLt0\x1d\x9e\xed\x87\xec\xc32\x1eN\x82e\x83_\xc3\x07"
    var_0 = maybe_0.get_or_else(bytes_0)
    var_1 = maybe_0.map(maybe_0)
    maybe_1 = module_0.Maybe(none_type_0, none_type_0)
    var_2 = maybe_1.to_validation()
    maybe_1.filter(var_2)


def test_case_5():
    str_0 = "\rLpRE R,!J:3u8"
    set_0 = {str_0}
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    maybe_0.map(set_0)


def test_case_6():
    dict_0 = {}
    bool_0 = False
    none_type_0 = None
    str_0 = "0gI\to\r\\l\rDZ"
    maybe_0 = module_0.Maybe(none_type_0, str_0)
    var_0 = maybe_0.bind(bool_0)
    var_1 = var_0.ap(dict_0)
    var_2 = var_1.to_box()
    var_2.to_box()


def test_case_7():
    bool_0 = False
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.to_validation()
    maybe_0.bind(var_0)


def test_case_8():
    bool_0 = False
    bool_1 = True
    complex_0 = -736 - 141j
    maybe_0 = module_0.Maybe(complex_0, complex_0)
    var_0 = maybe_0.ap(bool_1)
    bool_2 = False
    dict_0 = {bool_2: bool_2, bool_2: bool_2, bool_2: bool_2, bool_2: bool_2}
    maybe_1 = module_0.Maybe(dict_0, dict_0)
    var_1 = maybe_1.to_either()
    var_2 = var_1.bind(var_0)
    var_2.get_or_else(bool_0)


def test_case_9():
    bool_0 = False
    bool_1 = False
    maybe_0 = module_0.Maybe(bool_1, bool_1)
    maybe_0.ap(bool_0)


def test_case_10():
    int_0 = 1612
    none_type_0 = None
    bool_0 = True
    none_type_1 = None
    maybe_0 = module_0.Maybe(bool_0, none_type_1)
    var_0 = maybe_0.get_or_else(none_type_0)
    int_1 = 2735
    maybe_1 = module_0.Maybe(int_1, int_1)
    var_1 = maybe_1.filter(var_0)
    var_2 = var_1.filter(int_0)
    var_3 = maybe_1.to_box()


def test_case_11():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.to_validation()
    maybe_0.filter(var_0)


def test_case_12():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.to_validation()
    var_1 = maybe_0.get_or_else(var_0)
    var_2 = var_0.to_box()
    var_3 = var_2.to_either()


def test_case_13():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.get_or_else(none_type_0)
    var_1 = maybe_0.to_try()
    var_2 = maybe_0.to_try()
    var_3 = maybe_0.to_box()


def test_case_14():
    int_0 = -2000
    maybe_0 = module_0.Maybe(int_0, int_0)
    var_0 = maybe_0.to_either()
    var_1 = var_0.to_validation()
    var_1.to_validation()


def test_case_15():
    complex_0 = -1463.85509 + 413j
    none_type_0 = None
    maybe_0 = module_0.Maybe(complex_0, none_type_0)
    var_0 = maybe_0.to_either()
    var_1 = var_0.to_lazy()
    var_2 = var_1.to_try()
    complex_1 = -403.42 - 1063.04j
    bool_0 = False
    maybe_1 = module_0.Maybe(complex_1, bool_0)
    var_3 = maybe_1.to_validation()
    var_4 = maybe_1.to_lazy()
    var_4.get_or_else(complex_1)


def test_case_16():
    int_0 = -691
    bool_0 = True
    maybe_0 = module_0.Maybe(int_0, bool_0)
    var_0 = maybe_0.to_box()


def test_case_17():
    float_0 = -2104.19601
    maybe_0 = module_0.Maybe(float_0, float_0)
    var_0 = maybe_0.to_lazy()
    var_1 = maybe_0.get_or_else(float_0)
    var_2 = maybe_0.to_try()
    var_1.to_lazy()


def test_case_18():
    list_0 = []
    maybe_0 = module_0.Maybe(list_0, list_0)
    var_0 = maybe_0.to_lazy()
    var_1 = maybe_0.to_lazy()
    var_2 = var_1.to_validation()


def test_case_19():
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.ap(none_type_0)
    var_1 = maybe_0.map(var_0)
    var_2 = maybe_0.to_lazy()
    var_3 = maybe_0.get_or_else(maybe_0)
    var_4 = maybe_0.map(var_3)
    var_5 = var_4.to_validation()
    var_6 = var_3.to_lazy()
    var_7 = var_6.to_validation()
    var_8 = maybe_0.to_validation()


def test_case_20():
    bool_0 = False
    int_0 = -2595
    list_0 = []
    generic_0 = module_1.Generic()
    maybe_0 = module_0.Maybe(generic_0, generic_0)
    var_0 = maybe_0.ap(list_0)
    var_1 = var_0.to_either()
    var_2 = var_0.to_either()
    var_3 = maybe_0.get_or_else(maybe_0)
    var_4 = var_1.bind(list_0)
    var_5 = var_4.bind(int_0)
    bool_1 = maybe_0.__eq__(var_0)
    bool_2 = var_3.__eq__(list_0)
    var_6 = var_2.to_lazy()
    bool_3 = maybe_0.__eq__(bool_0)
    var_7 = var_1.bind(int_0)
    var_8 = var_3.to_lazy()
    var_9 = var_7.map(bool_0)
    var_10 = var_9.to_lazy()
    var_11 = var_1.to_box()
    var_12 = var_11.to_either()


def test_case_21():
    bool_0 = False
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    bool_1 = maybe_0.__eq__(maybe_0)
    maybe_0.map(bool_0)
