# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.maybe as module_0
import typing as module_1
import builtins as module_2


def test_case_0():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)


def test_case_1():
    bool_0 = False
    maybe_0 = module_0.Maybe(bool_0, bool_0)


def test_case_2():
    bytes_0 = b"\xdc\x87\xbch\x04\xb7\x8f\xa6\x18\xf3\xed\xcf"
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.filter(bytes_0)
    bool_1 = maybe_0.__eq__(maybe_0)
    var_1 = maybe_0.filter(bool_1)


def test_case_3():
    bytes_0 = b"\xdc\x87\xbch\x04\xb7\x8f\xa6\x18\xf3\xed\xcf"
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.filter(bytes_0)
    bool_1 = maybe_0.__eq__(bytes_0)


def test_case_4():
    bytes_0 = b"\xb4\xf8\x19\xc4$\x03\xd5\xc0"
    bool_0 = True
    maybe_0 = module_0.Maybe(bytes_0, bool_0)
    var_0 = maybe_0.map(maybe_0)
    var_1 = var_0.to_validation()
    var_2 = maybe_0.filter(var_1)
    bool_1 = var_0.__eq__(var_1)


def test_case_5():
    none_type_0 = None
    none_type_1 = None
    maybe_0 = module_0.Maybe(none_type_1, none_type_1)
    maybe_0.map(none_type_0)


def test_case_6():
    str_0 = " XLY0-F=)"
    none_type_0 = None
    none_type_1 = None
    maybe_0 = module_0.Maybe(str_0, str_0)
    var_0 = maybe_0.bind(none_type_1)
    var_1 = var_0.bind(none_type_0)
    var_2 = var_1.to_lazy()
    maybe_1 = module_0.Maybe(str_0, str_0)
    var_3 = maybe_1.to_box()
    var_4 = var_3.to_lazy()
    var_5 = maybe_1.filter(none_type_1)
    var_6 = var_1.to_validation()
    bool_0 = var_0.__eq__(none_type_1)


def test_case_7():
    float_0 = -2797.53
    dict_0 = {float_0: float_0, float_0: float_0, float_0: float_0}
    bool_0 = True
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.ap(float_0)
    var_1 = var_0.to_validation()
    var_2 = maybe_0.filter(var_1)
    bool_1 = var_2.__eq__(dict_0)


def test_case_8():
    generic_0 = module_1.Generic()
    bool_0 = False
    maybe_0 = module_0.Maybe(generic_0, bool_0)
    maybe_0.ap(bool_0)


def test_case_9():
    bool_0 = True
    bool_1 = False
    maybe_0 = module_0.Maybe(bool_0, bool_1)
    var_0 = maybe_0.get_or_else(bool_1)
    maybe_0.filter(var_0)


def test_case_10():
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.get_or_else(bool_0)
    bool_1 = maybe_0.__eq__(var_0)
    var_1 = maybe_0.get_or_else(var_0)
    var_0.get_or_else(none_type_0)


def test_case_11():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.to_either()


def test_case_12():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.to_try()
    var_1 = maybe_0.to_box()
    maybe_0.filter(var_1)


def test_case_13():
    float_0 = -2797.53
    dict_0 = {float_0: float_0, float_0: float_0, float_0: float_0}
    bool_0 = True
    maybe_0 = module_0.Maybe(float_0, bool_0)
    var_0 = maybe_0.to_lazy()
    bool_1 = var_0.__eq__(float_0)
    none_type_0 = None
    maybe_1 = module_0.Maybe(none_type_0, none_type_0)
    maybe_2 = module_0.Maybe(dict_0, float_0)
    var_1 = maybe_2.ap(float_0)
    var_2 = maybe_2.to_lazy()
    var_3 = var_2.to_either()


def test_case_14():
    none_type_0 = None
    bool_0 = False
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.to_lazy()


def test_case_15():
    bytes_0 = b"\xdc\x87\xbch\x1d\xb7\x8f\xa6\x18\xf3\xed\xcf"
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.filter(bytes_0)
    var_1 = maybe_0.to_try()
    bool_1 = maybe_0.__eq__(bytes_0)


def test_case_16():
    bool_0 = False
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.to_try()


def test_case_17():
    bytes_0 = b"\xdc\x87\xbch\x04\xb7\x8f\xa6\x18\xf3\xed\xcf"
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.filter(bytes_0)
    var_1 = maybe_0.to_validation()
    bool_1 = maybe_0.__eq__(bytes_0)


def test_case_18():
    set_0 = set()
    maybe_0 = module_0.Maybe(set_0, set_0)
    var_0 = maybe_0.to_validation()
    var_0.to_validation()


def test_case_19():
    none_type_0 = None
    none_type_0.to_lazy()


def test_case_20():
    float_0 = -1320.74188
    maybe_0 = module_0.Maybe(float_0, float_0)
    var_0 = maybe_0.to_lazy()
    var_1 = maybe_0.to_validation()
    var_2 = maybe_0.to_validation()
    var_3 = maybe_0.bind(var_2)
    var_4 = var_3.filter(maybe_0)
    var_5 = maybe_0.to_try()
    var_6 = var_4.to_try()
    var_7 = var_4.to_try()
    var_8 = var_5.bind(var_2)
    var_9 = maybe_0.to_lazy()
    var_10 = var_5.map(var_2)
    var_11 = maybe_0.bind(float_0)
    var_12 = maybe_0.to_validation()
    var_13 = maybe_0.bind(var_12)
    var_14 = var_13.map(var_12)
    var_15 = var_11.to_lazy()
    var_16 = var_12.to_try()
    bool_0 = maybe_0.__eq__(var_12)
    bool_1 = maybe_0.__eq__(var_14)
    var_17 = maybe_0.to_lazy()
    bool_2 = maybe_0.__eq__(maybe_0)
    var_17.get_or_else(float_0)


def test_case_21():
    float_0 = -2797.53
    bool_0 = True
    maybe_0 = module_0.Maybe(float_0, bool_0)
    var_0 = maybe_0.to_lazy()
    bool_1 = var_0.__eq__(float_0)
    none_type_0 = None
    maybe_1 = module_0.Maybe(none_type_0, none_type_0)
    maybe_2 = module_0.Maybe(var_0, bool_0)
    bool_2 = maybe_0.__eq__(none_type_0)
    var_1 = maybe_1.to_lazy()
    var_2 = var_1.to_try()
    var_3 = var_0.map(bool_1)
    bool_3 = False
    maybe_3 = module_0.Maybe(var_2, bool_3)
    float_1 = 486.401
    bool_4 = False
    maybe_4 = module_0.Maybe(float_1, bool_4)
    var_4 = maybe_1.to_validation()
    var_5 = maybe_1.to_either()
    var_6 = var_5.to_try()


def test_case_22():
    object_0 = module_2.object()
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    bool_0 = maybe_0.__eq__(object_0)
    var_0 = maybe_0.to_try()
    maybe_0.bind(var_0)


def test_case_23():
    none_type_0 = None
    bool_0 = False
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.to_lazy()
    bool_1 = maybe_0.__eq__(maybe_0)
    maybe_0.ap(none_type_0)
