# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.validation as module_0


def test_case_0():
    dict_0 = {}
    validation_0 = module_0.Validation(dict_0, dict_0)
    var_0 = validation_0.__eq__(validation_0)
    var_0.bind(validation_0)


def test_case_1():
    complex_0 = -410 - 1221j
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    var_0 = validation_0.__eq__(complex_0)
    var_0.to_either()


def test_case_2():
    bool_0 = False
    str_0 = '!VQtJB"s9'
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.to_maybe()
    var_1 = var_0.bind(str_0)
    var_2 = var_1.to_either()
    validation_1 = module_0.Validation(bool_0, str_0)
    var_3 = validation_1.__str__()
    var_3.is_success()


def test_case_3():
    dict_0 = {}
    validation_0 = module_0.Validation(dict_0, dict_0)
    var_0 = validation_0.to_either()
    var_1 = validation_0.__eq__(dict_0)
    var_1.map(var_1)


def test_case_4():
    none_type_0 = None
    none_type_0.to_try()


def test_case_5():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)


def test_case_6():
    bool_0 = True
    list_0 = [bool_0, bool_0, bool_0]
    validation_0 = module_0.Validation(list_0, bool_0)
    validation_0.__str__()


def test_case_7():
    bytes_0 = b"|w\xd6\xa2;"
    tuple_0 = (bytes_0,)
    validation_0 = module_0.Validation(tuple_0, bytes_0)
    var_0 = validation_0.is_fail()
    var_0.to_lazy()


def test_case_8():
    dict_0 = {}
    float_0 = -848.991598
    none_type_0 = None
    validation_0 = module_0.Validation(float_0, none_type_0)
    validation_0.map(dict_0)


def test_case_9():
    dict_0 = {}
    validation_0 = module_0.Validation(dict_0, dict_0)
    var_0 = validation_0.to_box()
    var_1 = var_0.to_lazy()
    var_2 = var_0.to_either()
    validation_0.bind(dict_0)


def test_case_10():
    none_type_0 = None
    int_0 = 426
    validation_0 = module_0.Validation(int_0, int_0)
    validation_0.ap(none_type_0)


def test_case_11():
    bytes_0 = b"\xdd(\xbc\x7f\x92\xde\x14\x89q"
    int_0 = -34
    validation_0 = module_0.Validation(int_0, int_0)
    var_0 = validation_0.to_lazy()
    var_1 = var_0.ap(bytes_0)
    var_1.to_either()


def test_case_12():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    bytes_0 = b"\x1bi\x152\xd4)\xba\xcb=\xe2\xb2\x8c"
    validation_1 = module_0.Validation(bytes_0, bytes_0)
    var_0 = validation_1.to_lazy()
    var_1 = var_0.to_try()
    var_1.to_maybe()


def test_case_13():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    validation_0.to_try()


def test_case_14():
    dict_0 = {}
    validation_0 = module_0.Validation(dict_0, dict_0)
    var_0 = validation_0.is_fail()
    var_1 = validation_0.to_lazy()
    var_2 = validation_0.__str__()
    var_3 = dict_0.__eq__(var_1)
    var_4 = validation_0.to_try()
    var_5 = validation_0.is_fail()
    var_6 = var_1.to_box()
    var_1.is_fail()


def test_case_15():
    str_0 = "3duo9E;ke"
    none_type_0 = None
    none_type_1 = None
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.to_either()
    var_1 = var_0.ap(str_0)
    var_2 = var_1.bind(str_0)
    var_3 = var_2.to_maybe()
    var_4 = var_3.__eq__(none_type_1)
    var_5 = var_4.__eq__(none_type_0)
    var_5.is_success()


def test_case_16():
    tuple_0 = ()
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, tuple_0)
    var_0 = validation_0.to_box()
    var_1 = validation_0.to_maybe()
    var_1.is_success()


def test_case_17():
    dict_0 = {}
    validation_0 = module_0.Validation(dict_0, dict_0)
    none_type_0 = None
    validation_1 = module_0.Validation(none_type_0, none_type_0)
    var_0 = validation_0.__eq__(validation_1)
    var_1 = validation_0.to_try()
    dict_0.is_fail()
