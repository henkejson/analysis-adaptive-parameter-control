# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typing as module_0
import pymonet.maybe as module_1


def test_case_0():
    generic_0 = module_0.Generic()
    maybe_0 = module_1.Maybe(generic_0, generic_0)


def test_case_1():
    bool_0 = False
    maybe_0 = module_1.Maybe(bool_0, bool_0)


def test_case_2():
    float_0 = -338.24835
    maybe_0 = module_1.Maybe(float_0, float_0)
    var_0 = maybe_0.to_lazy()
    var_1 = maybe_0.bind(var_0)
    bool_0 = maybe_0.__eq__(float_0)


def test_case_3():
    str_0 = "ia\x0bnX-a+<4"
    maybe_0 = module_1.Maybe(str_0, str_0)
    bool_0 = False
    var_0 = maybe_0.map(maybe_0)
    maybe_1 = module_1.Maybe(str_0, bool_0)
    var_1 = var_0.to_either()
    none_type_0 = None
    var_2 = var_0.to_box()
    var_3 = maybe_0.filter(none_type_0)
    var_4 = var_1.map(var_2)
    var_5 = maybe_0.bind(bool_0)
    maybe_1.filter(var_1)


def test_case_4():
    bool_0 = False
    dict_0 = {}
    tuple_0 = (bool_0, dict_0, dict_0)
    maybe_0 = module_1.Maybe(tuple_0, bool_0)
    var_0 = maybe_0.to_try()
    maybe_0.map(var_0)


def test_case_5():
    float_0 = 594.802
    none_type_0 = None
    maybe_0 = module_1.Maybe(float_0, float_0)
    var_0 = maybe_0.to_validation()
    var_1 = maybe_0.ap(none_type_0)
    var_2 = maybe_0.to_try()
    var_2.ap(none_type_0)


def test_case_6():
    bool_0 = True
    complex_0 = 2316.90517 + 1641.060195j
    bool_1 = False
    maybe_0 = module_1.Maybe(complex_0, bool_1)
    maybe_0.ap(bool_0)


def test_case_7():
    str_0 = "ia\x0bnX-a+<4"
    bool_0 = False
    none_type_0 = None
    bool_1 = True
    maybe_0 = module_1.Maybe(none_type_0, bool_1)
    var_0 = maybe_0.to_validation()
    maybe_1 = module_1.Maybe(none_type_0, maybe_0)
    var_1 = maybe_0.filter(var_0)
    var_2 = maybe_0.to_lazy()
    var_3 = maybe_0.ap(bool_0)
    maybe_2 = module_1.Maybe(none_type_0, bool_1)
    var_4 = maybe_2.get_or_else(str_0)
    var_5 = var_2.to_validation()
    maybe_3 = module_1.Maybe(var_5, var_3)
    var_6 = maybe_3.to_try()
    var_6.to_lazy()


def test_case_8():
    bytes_0 = b"\xd4\xbc\x12\x9b\xb7\xa4\xf7\xd5\xaeWt"
    bool_0 = True
    bytes_1 = b"\xacR\xb7[\x1f\x1a\xe2\xc6\x89\xb0\xcdy\xe6\xab\x92"
    none_type_0 = None
    maybe_0 = module_1.Maybe(bytes_1, none_type_0)
    var_0 = maybe_0.get_or_else(bool_0)
    var_0.ap(bytes_0)


def test_case_9():
    generic_0 = module_0.Generic()
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_1.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.to_either()
    var_1 = var_0.bind(generic_0)
    var_2 = var_1.to_box()
    var_3 = var_2.to_validation()


def test_case_10():
    none_type_0 = None
    maybe_0 = module_1.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.to_either()
    maybe_1 = module_1.Maybe(var_0, var_0)
    var_1 = maybe_0.to_either()
    maybe_0.bind(maybe_0)


def test_case_11():
    float_0 = -1438.012437
    int_0 = 1
    maybe_0 = module_1.Maybe(int_0, int_0)
    var_0 = maybe_0.to_box()
    var_0.filter(float_0)


def test_case_12():
    bool_0 = False
    maybe_0 = module_1.Maybe(bool_0, bool_0)
    maybe_1 = module_1.Maybe(bool_0, bool_0)
    bool_1 = maybe_1.__eq__(bool_0)
    var_0 = maybe_1.to_box()
    maybe_2 = module_1.Maybe(maybe_0, bool_0)
    var_1 = maybe_1.to_either()
    var_2 = maybe_1.to_either()
    var_2.ap(var_1)


def test_case_13():
    bool_0 = False
    maybe_0 = module_1.Maybe(bool_0, bool_0)
    var_0 = maybe_0.to_lazy()
    var_1 = var_0.to_either()
    var_2 = maybe_0.to_validation()
    maybe_1 = module_1.Maybe(bool_0, bool_0)
    var_3 = maybe_1.get_or_else(var_0)
    maybe_1.filter(var_0)


def test_case_14():
    float_0 = -338.24835
    maybe_0 = module_1.Maybe(float_0, float_0)
    var_0 = maybe_0.to_try()
    var_1 = maybe_0.bind(var_0)
    none_type_0 = None
    bool_0 = True
    maybe_1 = module_1.Maybe(none_type_0, bool_0)
    bool_1 = maybe_1.__eq__(float_0)


def test_case_15():
    int_0 = 2198
    bool_0 = False
    maybe_0 = module_1.Maybe(int_0, bool_0)
    var_0 = maybe_0.to_either()
    var_1 = var_0.to_box()
    var_2 = var_1.to_validation()
    var_3 = var_2.to_box()
    var_4 = maybe_0.to_either()
    var_5 = var_2.to_lazy()
    none_type_0 = None
    maybe_1 = module_1.Maybe(int_0, none_type_0)
    var_6 = maybe_1.to_validation()
    var_6.get_or_else(var_6)


def test_case_16():
    str_0 = "ia\x0bnX-a+<4"
    maybe_0 = module_1.Maybe(str_0, str_0)
    var_0 = maybe_0.ap(str_0)
    bool_0 = False
    var_1 = maybe_0.map(maybe_0)
    maybe_1 = module_1.Maybe(var_0, bool_0)
    var_2 = var_1.to_either()
    none_type_0 = None
    var_3 = var_0.to_lazy()
    var_4 = var_3.to_box()
    var_5 = maybe_1.to_try()
    var_6 = maybe_0.filter(none_type_0)
    var_7 = var_2.map(var_4)
    var_8 = maybe_0.bind(var_5)
    maybe_2 = module_1.Maybe(none_type_0, var_4)
    maybe_1.filter(var_2)


def test_case_17():
    str_0 = "ia\x0bnX-a+<4"
    maybe_0 = module_1.Maybe(str_0, str_0)
    var_0 = maybe_0.map(maybe_0)
    var_1 = maybe_0.map(var_0)
    bool_0 = var_1.__eq__(var_1)
    var_2 = maybe_0.filter(var_0)
    var_3 = var_2.to_lazy()
    var_4 = maybe_0.to_box()
    var_5 = var_3.bind(var_3)
    var_6 = var_2.filter(var_4)
    var_7 = var_3.ap(var_5)
    var_4.to_box()


def test_case_18():
    str_0 = "ia\x0bnX-a+<4"
    maybe_0 = module_1.Maybe(str_0, str_0)
    bool_0 = False
    var_0 = maybe_0.map(maybe_0)
    maybe_1 = module_1.Maybe(maybe_0, bool_0)
    var_1 = var_0.to_either()
    none_type_0 = None
    var_2 = maybe_0.to_lazy()
    var_3 = var_2.to_box()
    var_4 = maybe_1.to_try()
    var_5 = maybe_0.filter(none_type_0)
    maybe_2 = module_1.Maybe(var_3, var_3)
    var_6 = var_1.map(var_3)
    var_7 = var_5.to_box()
    var_8 = var_0.to_validation()
    maybe_3 = module_1.Maybe(none_type_0, var_3)
    var_9 = maybe_3.bind(var_8)
    bool_1 = maybe_1.__eq__(var_9)
    maybe_1.filter(var_8)


def test_case_19():
    str_0 = "ia\x0bnX-a+<4"
    maybe_0 = module_1.Maybe(str_0, str_0)
    var_0 = maybe_0.map(maybe_0)
    bool_0 = False
    var_1 = maybe_0.map(maybe_0)
    maybe_1 = module_1.Maybe(var_0, bool_0)
    var_2 = var_1.to_either()
    none_type_0 = None
    var_3 = var_0.to_lazy()
    var_4 = var_3.to_box()
    bool_1 = maybe_1.__eq__(maybe_1)
    var_5 = var_0.filter(bool_0)
    var_6 = var_2.ap(var_2)
    var_7 = var_0.to_box()
    var_8 = var_6.bind(var_7)
    maybe_2 = module_1.Maybe(none_type_0, var_7)
    var_9 = var_8.bind(maybe_0)
    var_10 = var_6.to_box()
    var_11 = var_2.ap(var_3)
    var_12 = var_3.bind(var_6)
    var_13 = var_6.to_validation()
    var_14 = var_7.to_lazy()
    var_15 = var_14.map(var_7)
    var_7.bind(var_13)
