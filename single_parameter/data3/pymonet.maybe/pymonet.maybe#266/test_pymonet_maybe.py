# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.maybe as module_0
import typing as module_1


def test_case_0():
    float_0 = 131.94
    maybe_0 = module_0.Maybe(float_0, float_0)


def test_case_1():
    bool_0 = False
    maybe_0 = module_0.Maybe(bool_0, bool_0)


def test_case_2():
    dict_0 = {}
    bool_0 = True
    maybe_0 = module_0.Maybe(dict_0, bool_0)
    var_0 = maybe_0.to_validation()
    maybe_1 = module_0.Maybe(dict_0, bool_0)
    bool_1 = maybe_0.__eq__(maybe_1)
    var_1 = maybe_1.map(maybe_1)
    var_2 = maybe_1.to_try()
    bool_2 = maybe_0.__eq__(maybe_1)
    var_3 = maybe_1.to_either()


def test_case_3():
    bool_0 = False
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.to_try()
    maybe_1 = module_0.Maybe(maybe_0, maybe_0)
    bool_1 = maybe_0.__eq__(maybe_1)
    var_1 = maybe_1.bind(bool_0)
    var_2 = maybe_1.filter(maybe_1)
    var_3 = var_1.to_try()
    var_4 = var_2.bind(var_0)
    var_5 = var_1.get_or_else(var_1)
    var_6 = var_3.get_or_else(bool_0)
    maybe_2 = module_0.Maybe(var_0, var_6)
    maybe_0.filter(bool_1)


def test_case_4():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    bool_0 = maybe_0.__eq__(maybe_0)
    var_0 = maybe_0.to_validation()
    var_1 = maybe_0.to_try()
    var_2 = maybe_0.get_or_else(maybe_0)
    maybe_0.map(var_0)


def test_case_5():
    str_0 = "Fx#m"
    maybe_0 = module_0.Maybe(str_0, str_0)
    var_0 = maybe_0.filter(maybe_0)
    var_1 = maybe_0.ap(str_0)
    var_2 = maybe_0.ap(str_0)
    var_3 = var_0.to_either()
    var_4 = maybe_0.bind(var_2)
    var_5 = maybe_0.ap(str_0)
    maybe_1 = module_0.Maybe(str_0, str_0)


def test_case_6():
    bool_0 = False
    float_0 = -3747.06341
    maybe_0 = module_0.Maybe(float_0, bool_0)
    maybe_1 = module_0.Maybe(maybe_0, maybe_0)
    var_0 = maybe_1.to_try()
    var_1 = maybe_0.ap(maybe_1)
    var_2 = maybe_1.bind(var_0)
    var_3 = maybe_1.to_lazy()
    var_4 = maybe_1.bind(var_0)
    var_5 = var_1.filter(var_1)
    var_6 = var_4.to_try()
    maybe_0.bind(var_1)


def test_case_7():
    int_0 = -65
    bool_0 = False
    maybe_0 = module_0.Maybe(int_0, bool_0)
    var_0 = maybe_0.to_try()
    bool_1 = var_0.__eq__(var_0)
    maybe_0.ap(var_0)


def test_case_8():
    int_0 = 5146
    bool_0 = True
    list_0 = [bool_0, int_0]
    maybe_0 = module_0.Maybe(bool_0, list_0)
    var_0 = maybe_0.to_lazy()
    var_1 = var_0.to_box()
    maybe_1 = module_0.Maybe(int_0, bool_0)
    var_2 = maybe_1.filter(bool_0)


def test_case_9():
    bool_0 = False
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.get_or_else(bool_0)
    var_1 = maybe_0.to_lazy()
    var_0.get_or_else(maybe_0)


def test_case_10():
    int_0 = 1096
    bool_0 = False
    maybe_0 = module_0.Maybe(int_0, bool_0)
    var_0 = maybe_0.to_either()
    var_0.get_or_else(bool_0)


def test_case_11():
    str_0 = "}B;k#3o"
    dict_0 = {}
    float_0 = 744.65499
    generic_0 = module_1.Generic(**dict_0)
    maybe_0 = module_0.Maybe(dict_0, generic_0)
    var_0 = maybe_0.filter(float_0)
    var_1 = var_0.to_lazy()
    var_2 = var_1.to_try()
    maybe_1 = module_0.Maybe(dict_0, var_2)
    var_3 = maybe_1.to_box()
    var_3.ap(str_0)


def test_case_12():
    none_type_0 = None
    bool_0 = False
    bool_1 = False
    maybe_0 = module_0.Maybe(bool_0, bool_1)
    var_0 = maybe_0.to_box()
    var_0.map(none_type_0)


def test_case_13():
    int_0 = 3441
    str_0 = "m\x0c<B@,SKvcp*f"
    bool_0 = True
    maybe_0 = module_0.Maybe(str_0, bool_0)
    var_0 = maybe_0.to_lazy()
    var_0.filter(int_0)


def test_case_14():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.to_lazy()
    var_1 = var_0.to_try()
    bool_0 = True
    maybe_1 = module_0.Maybe(none_type_0, bool_0)
    var_2 = maybe_1.to_validation()


def test_case_15():
    bool_0 = False
    bool_1 = True
    maybe_0 = module_0.Maybe(bool_0, bool_1)
    var_0 = maybe_0.to_lazy()
    var_1 = var_0.to_either()
    var_2 = var_1.to_validation()
    var_3 = var_2.to_lazy()
    tuple_0 = (bool_0, var_3, var_2)
    bool_2 = False
    maybe_1 = module_0.Maybe(tuple_0, bool_2)
    var_4 = maybe_1.to_validation()


def test_case_16():
    bool_0 = False
    none_type_0 = None
    maybe_0 = module_0.Maybe(bool_0, none_type_0)
    bool_1 = maybe_0.__eq__(none_type_0)
    bool_2 = True
    maybe_1 = module_0.Maybe(bool_0, bool_2)
    var_0 = maybe_1.get_or_else(maybe_0)
    bool_3 = False
    maybe_2 = module_0.Maybe(maybe_1, bool_3)
    bool_4 = var_0.__eq__(maybe_0)
    var_1 = var_0.to_validation()
    var_1.map(bool_0)


def test_case_17():
    bytes_0 = b"H|eQ\x07B\xfe0\xd4\x0f\xf1\xf0\xab\r\xe3J"
    list_0 = [bytes_0, bytes_0]
    none_type_0 = None
    maybe_0 = module_0.Maybe(list_0, none_type_0)
    bool_0 = True
    maybe_1 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_1.filter(bytes_0)
    maybe_0.filter(maybe_1)
