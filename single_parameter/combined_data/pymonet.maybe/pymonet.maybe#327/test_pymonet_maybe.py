# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.maybe as module_0


def test_case_0():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)


def test_case_1():
    bool_0 = False
    maybe_0 = module_0.Maybe(bool_0, bool_0)


def test_case_2():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    bool_0 = maybe_0.__eq__(maybe_0)
    maybe_0.ap(none_type_0)


def test_case_3():
    bytes_0 = b"v\x97`L"
    bool_0 = True
    maybe_0 = module_0.Maybe(bytes_0, bool_0)
    var_0 = maybe_0.get_or_else(maybe_0)
    bool_1 = maybe_0.__eq__(bytes_0)
    var_1 = maybe_0.to_validation()
    var_2 = var_0.ap(var_0)
    var_3 = maybe_0.filter(var_0)
    var_4 = var_2.filter(bool_0)


def test_case_4():
    int_0 = 3522
    str_0 = '\r++N5e4[Ag&Kb\r0qNy("'
    maybe_0 = module_0.Maybe(str_0, str_0)
    var_0 = maybe_0.map(int_0)
    float_0 = 3652.427
    maybe_1 = module_0.Maybe(float_0, float_0)
    var_1 = maybe_1.to_try()
    var_1.to_lazy()


def test_case_5():
    str_0 = "s2l)@2NQoe2d\nl"
    bool_0 = True
    maybe_0 = module_0.Maybe(str_0, bool_0)
    var_0 = maybe_0.to_box()
    var_1 = maybe_0.bind(maybe_0)
    maybe_1 = module_0.Maybe(str_0, var_0)
    var_2 = var_1.get_or_else(bool_0)
    bool_1 = var_1.__eq__(var_2)
    var_3 = maybe_1.to_validation()
    var_4 = maybe_1.ap(maybe_1)
    var_0.filter(var_4)


def test_case_6():
    int_0 = 947
    bool_0 = False
    maybe_0 = module_0.Maybe(int_0, bool_0)
    bool_1 = maybe_0.__eq__(int_0)
    var_0 = maybe_0.get_or_else(bool_0)
    maybe_0.bind(var_0)


def test_case_7():
    list_0 = []
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(list_0, bool_0)
    var_0 = maybe_0.filter(list_0)
    var_1 = var_0.filter(none_type_0)
    var_2 = var_1.ap(list_0)
    none_type_0.get_or_else(list_0)


def test_case_8():
    list_0 = []
    bool_0 = True
    maybe_0 = module_0.Maybe(list_0, bool_0)
    var_0 = maybe_0.filter(list_0)


def test_case_9():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    bool_0 = maybe_0.__eq__(maybe_0)
    maybe_0.filter(none_type_0)


def test_case_10():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    bool_0 = maybe_0.__eq__(none_type_0)
    maybe_1 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_1.get_or_else(maybe_1)
    maybe_1.filter(var_0)


def test_case_11():
    str_0 = "$X^_VG\x0bG"
    set_0 = {str_0, str_0, str_0, str_0}
    maybe_0 = module_0.Maybe(set_0, set_0)
    var_0 = maybe_0.to_either()
    var_1 = var_0.to_box()
    var_2 = var_1.to_validation()
    var_3 = var_2.to_lazy()
    var_0.filter(var_0)


def test_case_12():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.to_either()
    var_1 = var_0.to_try()
    var_1.to_either()


def test_case_13():
    str_0 = "\n        Applies the function inside the Maybe[A] structure to another applicative type for notempty Maybe.\n        For empty returns copy of itself\n\n        :param applicative: applicative contains function\n        :type applicative: Maybe[B]\n        :returns: new Maybe with result of contains function\n        :rtype: Maybe[A(B) | None]\n        "
    int_0 = 1835
    complex_0 = 2706 + 1806.7383j
    dict_0 = {int_0: int_0, int_0: int_0, int_0: complex_0, int_0: complex_0}
    maybe_0 = module_0.Maybe(dict_0, int_0)
    maybe_1 = module_0.Maybe(maybe_0, complex_0)
    var_0 = maybe_1.get_or_else(str_0)
    var_1 = maybe_1.to_box()
    var_0.to_validation()


def test_case_14():
    bytes_0 = b""
    bool_0 = False
    set_0 = {bytes_0, bool_0, bytes_0}
    maybe_0 = module_0.Maybe(set_0, bool_0)
    bytes_1 = b"\x07\xdb"
    none_type_0 = None
    bool_1 = False
    maybe_1 = module_0.Maybe(bytes_1, bool_1)
    var_0 = maybe_1.to_box()
    var_0.get_or_else(none_type_0)


def test_case_15():
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.to_validation()
    var_1 = maybe_0.to_lazy()
    var_2 = var_0.to_try()
    var_0.map(maybe_0)


def test_case_16():
    float_0 = -4015.21804
    dict_0 = {float_0: float_0, float_0: float_0, float_0: float_0, float_0: float_0}
    bool_0 = False
    maybe_0 = module_0.Maybe(dict_0, bool_0)
    var_0 = maybe_0.to_lazy()
    var_1 = maybe_0.to_validation()
    var_1.get_or_else(var_0)


def test_case_17():
    str_0 = "OOy1mj]\x0bLd]"
    bool_0 = True
    none_type_0 = None
    bool_1 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_1)
    maybe_1 = module_0.Maybe(str_0, bool_0)
    var_0 = maybe_0.get_or_else(none_type_0)
    var_1 = maybe_0.to_validation()
    bool_2 = maybe_0.__eq__(var_0)
    var_2 = maybe_1.ap(bool_2)
    var_3 = var_2.to_try()
    var_4 = maybe_1.get_or_else(var_1)


def test_case_18():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.to_try()


def test_case_19():
    str_0 = "s2l)@2NQoe2d\nl"
    bool_0 = True
    maybe_0 = module_0.Maybe(str_0, bool_0)
    var_0 = maybe_0.to_box()
    none_type_0 = None
    var_1 = maybe_0.bind(maybe_0)
    var_2 = maybe_0.to_lazy()
    float_0 = 73.2054
    var_3 = maybe_0.bind(float_0)
    var_4 = maybe_0.to_validation()
    var_5 = var_1.to_lazy()
    maybe_1 = module_0.Maybe(bool_0, var_4)
    var_6 = maybe_1.ap(var_2)
    var_7 = var_2.map(var_6)
    var_8 = maybe_0.ap(none_type_0)
    var_9 = maybe_0.filter(var_2)
    var_10 = maybe_0.ap(bool_0)
    maybe_2 = module_0.Maybe(var_10, bool_0)
    var_11 = maybe_2.to_either()
    var_12 = var_10.filter(var_11)
    var_13 = var_1.get_or_else(var_1)
    var_14 = var_10.to_lazy()
    var_15 = var_14.map(var_10)
    var_16 = var_8.filter(var_10)
    var_17 = maybe_0.map(var_10)
    var_18 = maybe_0.ap(maybe_0)
    maybe_3 = module_0.Maybe(var_10, str_0)
    var_19 = var_5.to_either()
    var_20 = var_8.get_or_else(var_10)
    var_15.to_either()


def test_case_20():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    bool_0 = maybe_0.__eq__(none_type_0)
    bytes_0 = b"\xeb\xde"
    maybe_1 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_1.to_lazy()
    var_1 = var_0.to_box()
    none_type_1 = None
    var_2 = maybe_1.to_validation()
    maybe_2 = module_0.Maybe(bytes_0, none_type_1)
    var_3 = maybe_2.to_either()
    var_4 = var_3.to_lazy()
    var_4.to_lazy()


def test_case_21():
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    bool_1 = maybe_0.__eq__(none_type_0)
    var_0 = maybe_0.to_validation()
    bool_2 = maybe_0.__eq__(maybe_0)
    var_1 = var_0.to_either()


def test_case_22():
    str_0 = "s2l)@2NQoe2d\nl"
    maybe_0 = module_0.Maybe(str_0, str_0)
    var_0 = maybe_0.map(str_0)
    bool_0 = maybe_0.__eq__(var_0)
    var_1 = var_0.to_validation()
    bool_1 = False
    maybe_1 = module_0.Maybe(str_0, bool_1)
    maybe_1.map(var_1)
