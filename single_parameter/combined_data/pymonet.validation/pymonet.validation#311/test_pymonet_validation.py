# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.validation as module_0


def test_case_0():
    bytes_0 = b"\xca:\x10\xe0|d\x05\xbf\x0b"
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    var_0 = validation_0.__eq__(bytes_0)
    var_0.is_fail()


def test_case_1():
    int_0 = 1034
    str_0 = "(1LI"
    validation_0 = module_0.Validation(int_0, str_0)
    var_0 = validation_0.to_either()
    float_0 = -1519.2309
    validation_1 = module_0.Validation(float_0, float_0)
    validation_1.is_success()


def test_case_2():
    str_0 = "iABY-1jT*M>"
    str_1 = "G./UB(V]?wM^`O!s"
    bytes_0 = b""
    set_0 = set()
    tuple_0 = (str_0, str_1, bytes_0, set_0)
    validation_0 = module_0.Validation(tuple_0, str_1)
    var_0 = validation_0.to_maybe()
    var_0.is_fail()


def test_case_3():
    int_0 = -1664
    validation_0 = module_0.Validation(int_0, int_0)


def test_case_4():
    int_0 = 1413
    validation_0 = module_0.Validation(int_0, int_0)
    validation_0.is_success()


def test_case_5():
    int_0 = -1664
    validation_0 = module_0.Validation(int_0, int_0)
    validation_0.is_fail()


def test_case_6():
    bool_0 = True
    float_0 = 875.47529
    validation_0 = module_0.Validation(float_0, float_0)
    validation_0.map(bool_0)


def test_case_7():
    bytes_0 = b"\x0eN\x1cI\x94<,"
    complex_0 = 1959.88 - 1115.963229j
    validation_0 = module_0.Validation(complex_0, complex_0)
    validation_0.bind(bytes_0)


def test_case_8():
    int_0 = -2643
    none_type_0 = None
    validation_0 = module_0.Validation(int_0, none_type_0)
    var_0 = validation_0.__eq__(none_type_0)
    validation_0.ap(int_0)


def test_case_9():
    float_0 = -96.28706
    validation_0 = module_0.Validation(float_0, float_0)
    var_0 = validation_0.to_box()
    var_0.to_box()


def test_case_10():
    complex_0 = -2021.09534 - 1354.00455j
    dict_0 = {complex_0: complex_0, complex_0: complex_0}
    validation_0 = module_0.Validation(complex_0, dict_0)
    validation_1 = module_0.Validation(dict_0, dict_0)
    var_0 = validation_0.to_lazy()
    var_1 = validation_1.to_box()
    var_2 = validation_1.to_maybe()
    validation_1.ap(var_1)


def test_case_11():
    bool_0 = True
    validation_0 = module_0.Validation(bool_0, bool_0)
    validation_0.to_try()


def test_case_12():
    bool_0 = False
    validation_0 = module_0.Validation(bool_0, bool_0)
    none_type_0 = None
    validation_1 = module_0.Validation(bool_0, none_type_0)
    var_0 = validation_1.__eq__(validation_0)
    validation_1.to_maybe()


def test_case_13():
    complex_0 = -2021.09534 - 1354.00455j
    dict_0 = {complex_0: complex_0, complex_0: complex_0}
    validation_0 = module_0.Validation(complex_0, dict_0)
    validation_1 = module_0.Validation(dict_0, dict_0)
    var_0 = validation_0.__eq__(validation_0)
    var_1 = validation_1.to_box()
    var_2 = validation_1.to_maybe()
    var_3 = validation_1.is_fail()
    var_4 = var_2.to_lazy()
    var_3.to_lazy()


def test_case_14():
    bytes_0 = b"\x16"
    none_type_0 = None
    bytes_1 = b"f\x14\xa9,]\x14-\xa7\x082z\xca"
    validation_0 = module_0.Validation(bytes_0, bytes_1)
    var_0 = validation_0.__str__()
    validation_1 = module_0.Validation(var_0, bytes_0)
    none_type_0.to_lazy()


def test_case_15():
    bytes_0 = b"\x16"
    bytes_1 = b"f\x14\xa9,]\x14-\xa7\x082z\xca"
    validation_0 = module_0.Validation(bytes_0, bytes_1)
    var_0 = validation_0.__str__()
    var_1 = validation_0.to_maybe()
    var_2 = validation_0.to_lazy()
    var_3 = var_2.to_box()
    var_4 = var_3.to_lazy()
    none_type_0 = None
    validation_1 = module_0.Validation(bytes_0, none_type_0)
    var_5 = var_4.__eq__(var_2)
    var_0.is_fail()


def test_case_16():
    complex_0 = -2021.09534 - 1354.00455j
    dict_0 = {}
    validation_0 = module_0.Validation(complex_0, dict_0)
    validation_1 = module_0.Validation(dict_0, dict_0)
    var_0 = validation_0.__eq__(validation_0)
    var_1 = validation_1.to_box()
    var_2 = validation_1.to_maybe()
    var_3 = validation_1.is_fail()
    var_4 = var_2.to_box()
    var_5 = validation_0.to_either()
    var_6 = validation_0.is_fail()
    var_7 = validation_1.__str__()
    var_8 = var_5.__str__()
    var_9 = var_4.to_try()
    var_9.is_fail()
