# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.maybe as module_0
import typing as module_1
import builtins as module_2


def test_case_0():
    str_0 = "\n    Data type for storage any type of data\n    "
    maybe_0 = module_0.Maybe(str_0, str_0)


def test_case_1():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)


def test_case_2():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.filter(bool_0)
    var_1 = var_0.get_or_else(maybe_0)
    bool_1 = var_0.__eq__(maybe_0)
    var_2 = var_0.to_validation()
    bool_2 = False
    maybe_1 = module_0.Maybe(var_2, bool_2)
    maybe_1.filter(var_2)


def test_case_3():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.filter(bool_0)
    bool_1 = var_0.__eq__(bool_0)
    var_1 = var_0.to_validation()
    bool_2 = False
    maybe_1 = module_0.Maybe(var_1, bool_2)
    maybe_1.filter(var_1)


def test_case_4():
    float_0 = -2456.999
    maybe_0 = module_0.Maybe(float_0, float_0)
    var_0 = maybe_0.to_either()
    int_0 = 6172
    bool_0 = False
    set_0 = {int_0, bool_0}
    maybe_1 = module_0.Maybe(int_0, int_0)
    maybe_2 = module_0.Maybe(set_0, set_0)
    var_1 = maybe_1.map(maybe_2)
    var_2 = maybe_1.filter(bool_0)
    var_3 = var_2.get_or_else(set_0)
    var_3.filter(bool_0)


def test_case_5():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.to_validation()
    var_1 = maybe_0.to_box()
    maybe_0.map(var_1)


def test_case_6():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.filter(bool_0)
    var_1 = var_0.bind(var_0)
    bool_1 = False
    maybe_1 = module_0.Maybe(var_1, bool_1)
    maybe_1.filter(var_1)


def test_case_7():
    bool_0 = False
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.to_validation()
    maybe_0.bind(var_0)


def test_case_8():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.filter(bool_0)
    var_1 = var_0.get_or_else(maybe_0)
    var_2 = var_1.ap(var_1)
    var_3 = var_0.to_validation()
    bool_1 = False
    maybe_1 = module_0.Maybe(var_3, bool_1)
    maybe_1.filter(var_3)


def test_case_9():
    none_type_0 = None
    tuple_0 = ()
    maybe_0 = module_0.Maybe(tuple_0, tuple_0)
    maybe_0.ap(none_type_0)


def test_case_10():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.filter(bool_0)
    bool_1 = False
    maybe_1 = module_0.Maybe(bool_0, bool_1)
    maybe_1.filter(bool_1)


def test_case_11():
    generic_0 = module_1.Generic()
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.get_or_else(generic_0)
    var_0.to_validation()


def test_case_12():
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.to_either()
    var_1 = var_0.to_try()
    var_1.to_either()


def test_case_13():
    object_0 = module_2.object()
    none_type_0 = None
    maybe_0 = module_0.Maybe(object_0, none_type_0)
    none_type_1 = None
    var_0 = maybe_0.get_or_else(none_type_1)
    var_1 = maybe_0.to_lazy()
    var_2 = maybe_0.to_either()
    none_type_2 = None
    maybe_0.filter(none_type_2)


def test_case_14():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.filter(bool_0)
    var_1 = var_0.to_box()
    var_2 = var_0.to_validation()
    bool_1 = False
    maybe_1 = module_0.Maybe(var_2, bool_1)
    maybe_1.filter(var_2)


def test_case_15():
    int_0 = 0
    maybe_0 = module_0.Maybe(int_0, int_0)
    var_0 = maybe_0.to_box()
    maybe_1 = module_0.Maybe(var_0, int_0)
    var_0.map(var_0)


def test_case_16():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.to_lazy()
    var_1 = maybe_0.filter(bool_0)
    var_2 = var_1.get_or_else(maybe_0)
    var_3 = var_1.to_validation()
    bool_1 = False
    maybe_1 = module_0.Maybe(var_3, bool_1)
    maybe_1.filter(var_3)


def test_case_17():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.to_lazy()
    var_0.filter(none_type_0)


def test_case_18():
    str_0 = "UDDGY{_<+S8P=A"
    maybe_0 = module_0.Maybe(str_0, str_0)
    var_0 = maybe_0.to_try()
    var_0.to_lazy()


def test_case_19():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.to_try()


def test_case_20():
    complex_0 = -916.0462 + 1690.3j
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.to_validation()
    var_0.bind(complex_0)


def test_case_21():
    bool_0 = False
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.to_validation()
    var_1 = var_0.to_either()


def test_case_22():
    int_0 = 1689
    maybe_0 = module_0.Maybe(int_0, int_0)
    var_0 = maybe_0.to_box()
    bool_0 = True
    maybe_1 = module_0.Maybe(bool_0, bool_0)
    var_1 = maybe_1.to_lazy()
    var_2 = var_1.to_either()
    var_3 = var_1.to_either()
    var_4 = var_2.to_validation()
    none_type_0 = None
    var_4.ap(none_type_0)


def test_case_23():
    int_0 = 1
    bool_0 = True
    bool_1 = True
    maybe_0 = module_0.Maybe(bool_0, bool_1)
    var_0 = maybe_0.filter(int_0)
    var_1 = var_0.to_validation()
    none_type_0 = None
    bool_2 = False
    maybe_1 = module_0.Maybe(none_type_0, bool_2)
    var_2 = maybe_1.get_or_else(maybe_1)
    bytes_0 = b"\xe2\xec\xbe\x0e}_\xd4\xf7"
    set_0 = {bytes_0, bytes_0, bytes_0, bytes_0}
    bool_3 = False
    maybe_2 = module_0.Maybe(set_0, bool_3)
    maybe_2.filter(var_2)


def test_case_24():
    bool_0 = True
    bool_1 = False
    maybe_0 = module_0.Maybe(bool_0, bool_1)
    var_0 = maybe_0.to_lazy()
    var_1 = var_0.to_validation()
    maybe_1 = module_0.Maybe(bool_0, bool_0)
    var_2 = maybe_1.filter(bool_0)
    var_3 = var_2.to_validation()
    bool_2 = False
    maybe_2 = module_0.Maybe(var_2, bool_2)
    maybe_2.filter(var_3)


def test_case_25():
    int_0 = 1
    bool_0 = True
    bool_1 = True
    maybe_0 = module_0.Maybe(bool_0, bool_1)
    var_0 = maybe_0.filter(int_0)
    var_1 = var_0.to_validation()
    none_type_0 = None
    bool_2 = False
    maybe_1 = module_0.Maybe(none_type_0, bool_2)
    bool_3 = maybe_0.__eq__(maybe_1)
    var_2 = maybe_1.to_validation()
    bytes_0 = b"\xe2\xec\xbe\x0e}_\xd4\xf7"
    set_0 = {bytes_0, bytes_0, bytes_0, bytes_0}
    bool_4 = False
    maybe_2 = module_0.Maybe(set_0, bool_4)
    maybe_2.filter(var_2)


def test_case_26():
    bool_0 = False
    none_type_0 = None
    maybe_0 = module_0.Maybe(bool_0, none_type_0)
    none_type_1 = None
    maybe_1 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_1.to_validation()
    maybe_2 = module_0.Maybe(none_type_1, none_type_1)
    var_1 = maybe_2.to_box()
    var_2 = maybe_2.to_box()
    bool_1 = maybe_2.__eq__(maybe_2)
    var_3 = maybe_2.to_box()
    var_4 = var_3.to_lazy()
    var_3.get_or_else(none_type_1)
