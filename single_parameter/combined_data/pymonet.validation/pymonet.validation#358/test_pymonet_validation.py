# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.validation as module_0


def test_case_0():
    bool_0 = True
    int_0 = 346
    validation_0 = module_0.Validation(int_0, int_0)
    var_0 = validation_0.__eq__(bool_0)
    var_0.to_either()


def test_case_1():
    bytes_0 = b"\x9e*Y\xec\x10\xb7"
    validation_0 = module_0.Validation(bytes_0, bytes_0)
    var_0 = validation_0.to_either()
    var_1 = validation_0.to_maybe()
    var_2 = validation_0.__str__()
    var_3 = var_0.bind(var_2)
    var_4 = validation_0.__eq__(var_1)
    var_5 = var_0.ap(var_2)
    var_0.is_fail()


def test_case_2():
    bytes_0 = b""
    validation_0 = module_0.Validation(bytes_0, bytes_0)
    var_0 = validation_0.to_either()
    var_1 = var_0.to_maybe()
    var_2 = var_1.to_box()


def test_case_3():
    bytes_0 = b"\x9e*Y\xec\x10\xb7"
    validation_0 = module_0.Validation(bytes_0, bytes_0)
    var_0 = validation_0.to_either()
    var_1 = validation_0.to_maybe()
    var_2 = validation_0.to_maybe()
    var_3 = var_0.bind(var_2)
    var_4 = validation_0.__eq__(var_1)
    var_5 = var_0.ap(var_2)
    validation_0.bind(var_2)


def test_case_4():
    bytes_0 = b"\x9e*Y\xec\x10\xb7"
    validation_0 = module_0.Validation(bytes_0, bytes_0)
    var_0 = validation_0.to_maybe()
    validation_0.bind(validation_0)


def test_case_5():
    complex_0 = -920.214152601509 - 586j
    complex_0.to_lazy()


def test_case_6():
    complex_0 = -920.214152601509 - 561.9296446619186j
    var_0 = module_0.Validation(complex_0, complex_0)


def test_case_7():
    tuple_0 = ()
    none_type_0 = None
    validation_0 = module_0.Validation(tuple_0, none_type_0)
    validation_0.is_fail()


def test_case_8():
    bool_0 = False
    bool_1 = False
    validation_0 = module_0.Validation(bool_1, bool_1)
    validation_0.map(bool_0)


def test_case_9():
    bytes_0 = b"\x9e*Y\xec\x10\xb7"
    validation_0 = module_0.Validation(bytes_0, bytes_0)
    var_0 = validation_0.to_either()
    var_1 = validation_0.to_maybe()
    validation_0.ap(var_0)


def test_case_10():
    float_0 = -2145.203
    dict_0 = {}
    validation_0 = module_0.Validation(dict_0, float_0)
    var_0 = validation_0.to_box()
    var_1 = var_0.__eq__(dict_0)
    var_1.is_success()


def test_case_11():
    bool_0 = True
    validation_0 = module_0.Validation(bool_0, bool_0)
    var_0 = validation_0.to_lazy()
    var_1 = var_0.to_try()
    var_1.bind(bool_0)


def test_case_12():
    bool_0 = True
    validation_0 = module_0.Validation(bool_0, bool_0)
    validation_0.to_try()


def test_case_13():
    bytes_0 = b"\x9e*Y\xec\x10\xb7"
    validation_0 = module_0.Validation(bytes_0, bytes_0)
    var_0 = validation_0.to_either()
    var_1 = var_0.ap(var_0)
    var_2 = var_0.bind(bytes_0)
    complex_0 = -920.214152601509 - 586j
    var_3 = validation_0.__eq__(validation_0)
    var_4 = var_0.__eq__(var_3)
    var_5 = var_1.ap(complex_0)
    var_6 = var_5.bind(var_5)


def test_case_14():
    tuple_0 = ()
    validation_0 = module_0.Validation(tuple_0, tuple_0)
    none_type_0 = None
    list_0 = [validation_0]
    validation_1 = module_0.Validation(list_0, none_type_0)
    var_0 = validation_0.is_fail()
    var_1 = validation_0.__str__()
    validation_1.is_fail()


def test_case_15():
    bytes_0 = b"\x9e*Y\xec\x10\xb7"
    validation_0 = module_0.Validation(bytes_0, bytes_0)
    var_0 = module_0.Validation(bytes_0, validation_0)
    var_1 = var_0.__eq__(validation_0)
    var_1.to_maybe()


def test_case_16():
    set_0 = set()
    validation_0 = module_0.Validation(set_0, set_0)
    var_0 = validation_0.to_maybe()
    var_0.map(set_0)
