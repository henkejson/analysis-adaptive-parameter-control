# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.maybe as module_0


def test_case_0():
    str_0 = "Try[value={}, is_success={}]"
    maybe_0 = module_0.Maybe(str_0, str_0)


def test_case_1():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)


def test_case_2():
    bytes_0 = b"FPg\xca\xda\x9b\x83\xb2"
    maybe_0 = module_0.Maybe(bytes_0, bytes_0)
    var_0 = maybe_0.bind(bytes_0)
    var_1 = maybe_0.to_box()
    var_2 = maybe_0.bind(var_1)
    bool_0 = var_0.__eq__(var_2)
    none_type_0 = None
    var_1.get_or_else(none_type_0)


def test_case_3():
    str_0 = "99in&%V[}bs7'>z7"
    str_1 = ":9Tt"
    float_0 = -261.0
    maybe_0 = module_0.Maybe(str_1, float_0)
    none_type_0 = None
    bool_0 = maybe_0.__eq__(none_type_0)
    var_0 = maybe_0.map(str_0)


def test_case_4():
    float_0 = -1014.8092
    bool_0 = True
    maybe_0 = module_0.Maybe(float_0, bool_0)
    var_0 = maybe_0.to_lazy()
    tuple_0 = (float_0, var_0, maybe_0, var_0)
    bool_1 = True
    maybe_1 = module_0.Maybe(tuple_0, bool_1)
    str_0 = "X)_K Z<!K=L#(},Rd"
    bool_2 = False
    maybe_2 = module_0.Maybe(str_0, bool_2)
    maybe_2.map(maybe_1)


def test_case_5():
    str_0 = "99in&%V[}bs7'>z7"
    str_1 = ":9Tt"
    float_0 = -261.0
    maybe_0 = module_0.Maybe(str_1, float_0)
    var_0 = maybe_0.to_try()
    var_1 = maybe_0.bind(str_1)
    var_2 = var_1.filter(var_0)
    bool_0 = False
    maybe_1 = module_0.Maybe(maybe_0, bool_0)
    maybe_2 = module_0.Maybe(str_0, var_0)


def test_case_6():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.get_or_else(bool_0)
    maybe_1 = module_0.Maybe(var_0, bool_0)
    maybe_2 = module_0.Maybe(var_0, maybe_0)
    bool_1 = maybe_0.__eq__(bool_0)
    var_1 = maybe_0.to_validation()
    var_2 = maybe_2.bind(var_0)
    int_0 = 0
    maybe_3 = module_0.Maybe(bool_0, int_0)
    none_type_0 = None
    var_3 = var_2.to_try()
    var_4 = maybe_0.get_or_else(none_type_0)
    var_5 = maybe_2.to_validation()
    maybe_4 = module_0.Maybe(bool_0, int_0)
    var_6 = maybe_4.to_validation()
    var_7 = maybe_4.get_or_else(none_type_0)
    var_8 = var_6.to_lazy()
    var_9 = maybe_3.to_box()
    maybe_3.bind(var_9)


def test_case_7():
    str_0 = "mQy2<R#"
    set_0 = set()
    maybe_0 = module_0.Maybe(str_0, str_0)
    var_0 = maybe_0.filter(str_0)
    var_1 = var_0.ap(set_0)
    var_2 = var_1.get_or_else(str_0)
    list_0 = [str_0, var_2, var_1, str_0]
    maybe_1 = module_0.Maybe(list_0, var_2)
    var_3 = maybe_1.to_either()
    var_4 = var_3.to_validation()


def test_case_8():
    bool_0 = False
    set_0 = {bool_0, bool_0}
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    maybe_0.ap(set_0)


def test_case_9():
    float_0 = -2043.419195
    int_0 = 1
    none_type_0 = None
    maybe_0 = module_0.Maybe(int_0, none_type_0)
    maybe_0.filter(float_0)


def test_case_10():
    str_0 = "99in&%V[}bs7'>z7"
    str_1 = ":9Tt"
    float_0 = -261.0
    maybe_0 = module_0.Maybe(str_1, float_0)
    none_type_0 = None
    bool_0 = maybe_0.__eq__(none_type_0)
    var_0 = maybe_0.map(str_0)
    var_1 = var_0.get_or_else(none_type_0)


def test_case_11():
    int_0 = 1
    bool_0 = False
    maybe_0 = module_0.Maybe(int_0, bool_0)
    maybe_1 = module_0.Maybe(maybe_0, bool_0)
    var_0 = maybe_1.to_either()
    var_1 = maybe_1.get_or_else(maybe_1)
    var_2 = var_0.to_lazy()


def test_case_12():
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.to_either()


def test_case_13():
    float_0 = 215.83
    complex_0 = 2189 - 3133.3666j
    bool_0 = False
    maybe_0 = module_0.Maybe(complex_0, complex_0)
    var_0 = maybe_0.to_validation()
    var_1 = maybe_0.to_validation()
    var_2 = maybe_0.map(var_0)
    var_3 = var_0.to_box()
    var_4 = maybe_0.to_box()
    var_5 = var_1.to_either()
    var_6 = var_3.to_either()
    bool_1 = var_3.__eq__(complex_0)
    var_7 = var_6.to_validation()
    maybe_1 = module_0.Maybe(complex_0, bool_0)
    var_8 = maybe_1.to_box()
    var_8.bind(float_0)


def test_case_14():
    bool_0 = False
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    maybe_1 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_1.to_lazy()
    var_0.to_lazy()


def test_case_15():
    set_0 = set()
    maybe_0 = module_0.Maybe(set_0, set_0)
    var_0 = maybe_0.to_try()
    bool_0 = maybe_0.__eq__(set_0)
    maybe_1 = module_0.Maybe(var_0, set_0)
    var_1 = maybe_1.to_try()
    bool_1 = var_0.__eq__(var_1)
    var_1.map(var_0)


def test_case_16():
    str_0 = "99in&%V[}bs7'>z7"
    str_1 = ":9Tt"
    float_0 = -261.0
    maybe_0 = module_0.Maybe(str_1, float_0)
    none_type_0 = None
    bool_0 = maybe_0.__eq__(none_type_0)
    var_0 = maybe_0.to_either()
    var_1 = maybe_0.to_validation()
    bool_1 = var_1.__eq__(str_0)


def test_case_17():
    float_0 = 687.0974
    none_type_0 = None
    maybe_0 = module_0.Maybe(float_0, none_type_0)
    var_0 = maybe_0.to_validation()


def test_case_18():
    str_0 = "99in&%V[}bs7'>z7"
    str_1 = "u"
    maybe_0 = module_0.Maybe(str_0, str_1)
    var_0 = maybe_0.to_try()
    bool_0 = maybe_0.__eq__(str_1)
    none_type_0 = None
    maybe_1 = module_0.Maybe(none_type_0, str_0)
    var_1 = maybe_0.to_try()
    var_2 = maybe_0.to_lazy()
    var_3 = var_2.to_box()
    var_4 = maybe_0.filter(var_3)
    maybe_2 = module_0.Maybe(bool_0, var_3)
    maybe_3 = module_0.Maybe(maybe_0, none_type_0)


def test_case_19():
    dict_0 = {}
    none_type_0 = None
    list_0 = [none_type_0, dict_0, dict_0, dict_0]
    maybe_0 = module_0.Maybe(none_type_0, list_0)
    var_0 = maybe_0.to_try()
    none_type_1 = None
    bool_0 = False
    maybe_1 = module_0.Maybe(none_type_1, bool_0)
    var_1 = maybe_0.ap(maybe_0)
    var_2 = maybe_0.filter(var_0)
    var_3 = maybe_1.to_lazy()
    var_4 = maybe_1.to_either()
    var_5 = maybe_1.get_or_else(var_3)
    var_6 = var_3.to_box()
    var_5.to_try()


def test_case_20():
    bytes_0 = b"\xb5\xfd\xec\xbe\xb0=WH[\xd0\xcb"
    none_type_0 = None
    none_type_1 = None
    maybe_0 = module_0.Maybe(none_type_1, bytes_0)
    var_0 = maybe_0.map(maybe_0)
    var_1 = var_0.filter(none_type_1)
    bool_0 = maybe_0.__eq__(maybe_0)
    bool_1 = False
    var_2 = var_0.bind(bool_1)
    var_3 = var_0.to_try()
    var_4 = var_0.to_either()
    maybe_1 = module_0.Maybe(bytes_0, none_type_0)
    bool_2 = var_2.__eq__(maybe_1)
    maybe_2 = module_0.Maybe(bool_2, bool_2)
    var_5 = var_2.to_validation()
    maybe_2.filter(none_type_0)


def test_case_21():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    bool_0 = maybe_0.__eq__(maybe_0)
    var_0 = maybe_0.to_box()
    maybe_0.bind(var_0)
