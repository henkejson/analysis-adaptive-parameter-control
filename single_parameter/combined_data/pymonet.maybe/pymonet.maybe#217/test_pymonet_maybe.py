# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.maybe as module_0


def test_case_0():
    int_0 = 3549
    maybe_0 = module_0.Maybe(int_0, int_0)


def test_case_1():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)


def test_case_2():
    complex_0 = -2518.1 - 1649.556472j
    none_type_0 = None
    maybe_0 = module_0.Maybe(complex_0, complex_0)
    bool_0 = maybe_0.__eq__(maybe_0)
    var_0 = maybe_0.map(complex_0)
    var_1 = var_0.get_or_else(complex_0)
    var_1.get_or_else(none_type_0)


def test_case_3():
    complex_0 = -2518.1 - 1649.556472j
    maybe_0 = module_0.Maybe(complex_0, complex_0)
    var_0 = maybe_0.filter(complex_0)
    bool_0 = maybe_0.__eq__(complex_0)
    maybe_1 = module_0.Maybe(var_0, bool_0)
    var_1 = maybe_1.to_either()
    maybe_1.filter(complex_0)


def test_case_4():
    dict_0 = {}
    maybe_0 = module_0.Maybe(dict_0, dict_0)
    maybe_0.ap(maybe_0)


def test_case_5():
    complex_0 = -2518.1 - 1649.556472j
    maybe_0 = module_0.Maybe(complex_0, complex_0)
    var_0 = maybe_0.filter(complex_0)
    bool_0 = var_0.__eq__(maybe_0)
    var_1 = var_0.ap(complex_0)
    var_2 = maybe_0.map(complex_0)
    var_3 = var_2.bind(var_0)
    maybe_1 = module_0.Maybe(var_3, bool_0)
    var_4 = maybe_1.to_either()
    maybe_2 = module_0.Maybe(maybe_1, var_4)
    var_5 = var_4.to_try()
    maybe_1.filter(var_5)


def test_case_6():
    bool_0 = False
    dict_0 = {bool_0: bool_0, bool_0: bool_0, bool_0: bool_0, bool_0: bool_0}
    maybe_0 = module_0.Maybe(dict_0, bool_0)
    none_type_0 = None
    bool_1 = maybe_0.__eq__(none_type_0)
    bool_2 = maybe_0.__eq__(dict_0)
    var_0 = maybe_0.to_box()
    maybe_0.bind(var_0)


def test_case_7():
    complex_0 = -2518.1 - 1649.556472j
    maybe_0 = module_0.Maybe(complex_0, complex_0)
    var_0 = maybe_0.filter(complex_0)
    bool_0 = maybe_0.__eq__(complex_0)
    var_1 = var_0.ap(complex_0)
    var_2 = maybe_0.map(complex_0)
    maybe_1 = module_0.Maybe(var_0, bool_0)
    var_3 = maybe_1.to_either()
    maybe_1.filter(complex_0)


def test_case_8():
    bytes_0 = b"N\x0c\xe3G\xc57\x92b\xb4\xb8e\n\xe7)\x14\x9ch\x1f\x01\xd2"
    bool_0 = False
    maybe_0 = module_0.Maybe(bytes_0, bool_0)
    var_0 = maybe_0.to_validation()
    maybe_0.ap(bytes_0)


def test_case_9():
    str_0 = "^\x0bnV-<&{BWo5;u+G"
    maybe_0 = module_0.Maybe(str_0, str_0)
    var_0 = maybe_0.filter(maybe_0)
    var_1 = maybe_0.to_either()
    var_2 = var_1.map(var_1)
    bool_0 = var_1.__eq__(var_1)
    bool_0.to_lazy()


def test_case_10():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.get_or_else(none_type_0)
    maybe_1 = module_0.Maybe(var_0, var_0)
    var_1 = maybe_0.to_try()
    var_2 = maybe_1.to_validation()
    var_0.to_either()


def test_case_11():
    int_0 = 1
    bool_0 = True
    maybe_0 = module_0.Maybe(int_0, bool_0)
    var_0 = maybe_0.to_box()
    var_1 = var_0.to_lazy()
    var_2 = var_1.to_either()
    var_2.to_either()


def test_case_12():
    int_0 = -1930
    bool_0 = True
    maybe_0 = module_0.Maybe(int_0, bool_0)
    var_0 = maybe_0.to_lazy()
    var_1 = var_0.to_try()
    var_1.to_try()


def test_case_13():
    str_0 = "459"
    none_type_0 = None
    bool_0 = False
    maybe_0 = module_0.Maybe(str_0, bool_0)
    var_0 = maybe_0.to_lazy()
    var_1 = maybe_0.get_or_else(none_type_0)
    var_1.to_try()


def test_case_14():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.to_try()
    var_0.to_box()


def test_case_15():
    none_type_0 = None
    bool_0 = True
    set_0 = {bool_0, bool_0, bool_0}
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.to_lazy()
    var_1 = var_0.to_validation()
    maybe_1 = module_0.Maybe(set_0, set_0)
    var_2 = maybe_1.get_or_else(none_type_0)
    var_3 = maybe_1.to_validation()
    maybe_2 = module_0.Maybe(var_3, bool_0)
    var_4 = maybe_1.map(var_3)
    var_5 = maybe_2.to_box()
    var_5.get_or_else(none_type_0)


def test_case_16():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    bool_0 = maybe_0.__eq__(maybe_0)
    maybe_0.map(bool_0)


def test_case_17():
    complex_0 = -2518.1 - 1649.556472j
    maybe_0 = module_0.Maybe(complex_0, complex_0)
    var_0 = maybe_0.filter(complex_0)
    bool_0 = var_0.__eq__(maybe_0)
    var_1 = var_0.ap(complex_0)
    var_2 = maybe_0.map(complex_0)
    maybe_1 = module_0.Maybe(var_0, bool_0)
    var_3 = maybe_1.to_either()
    maybe_2 = module_0.Maybe(maybe_1, var_3)
    var_4 = var_3.to_try()
    maybe_1.filter(var_4)
