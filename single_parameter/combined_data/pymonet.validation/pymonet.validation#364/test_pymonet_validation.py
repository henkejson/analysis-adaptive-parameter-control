# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.validation as module_0


def test_case_0():
    bool_0 = True
    validation_0 = module_0.Validation(bool_0, bool_0)
    var_0 = validation_0.__eq__(validation_0)
    var_1 = validation_0.__eq__(bool_0)
    var_0.is_success()


def test_case_1():
    str_0 = " x7!pUGZ'Jn\r>)k{"
    bytes_0 = b"\x0c-\xcc\xcd\xff\x1c\x0e\xd6TM\xde\xc9c\x12$;\xa8\x8cD"
    tuple_0 = ()
    int_0 = 1
    tuple_1 = (bytes_0, tuple_0, int_0, int_0)
    tuple_2 = (tuple_1,)
    bool_0 = True
    dict_0 = {tuple_2: bool_0}
    validation_0 = module_0.Validation(dict_0, tuple_1)
    var_0 = validation_0.__str__()
    var_1 = var_0.__str__()
    var_1.map(str_0)


def test_case_2():
    str_0 = "y#"
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.to_either()
    var_1 = validation_0.to_try()
    var_1.to_lazy()


def test_case_3():
    bytes_0 = b"E\xe31E\x03\x84\x85\xb9:\x9b\xc4\xaf\xbcY"
    validation_0 = module_0.Validation(bytes_0, bytes_0)
    var_0 = validation_0.to_maybe()


def test_case_4():
    bytes_0 = b"E\xe31E\x03\x84\x85\xb9:\x9b\xc4\xaf\xbcY"
    bytes_0.to_box()


def test_case_5():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)


def test_case_6():
    float_0 = 3110.0
    validation_0 = module_0.Validation(float_0, float_0)
    validation_0.is_fail()


def test_case_7():
    none_type_0 = None
    bytes_0 = b"\xd3"
    bool_0 = False
    tuple_0 = (bool_0,)
    list_0 = [bool_0, bytes_0, tuple_0, bool_0]
    tuple_1 = (bytes_0, tuple_0, list_0)
    validation_0 = module_0.Validation(tuple_1, tuple_1)
    var_0 = validation_0.to_box()
    validation_0.map(none_type_0)


def test_case_8():
    bool_0 = False
    validation_0 = module_0.Validation(bool_0, bool_0)
    var_0 = validation_0.__eq__(validation_0)
    validation_0.bind(validation_0)


def test_case_9():
    int_0 = 1303
    validation_0 = module_0.Validation(int_0, int_0)
    validation_0.ap(validation_0)


def test_case_10():
    bytes_0 = b"E\xe31E\x03\x84\x85\xb9:\x9b\xc4\xaf\xbcY"
    validation_0 = module_0.Validation(bytes_0, bytes_0)
    var_0 = validation_0.to_maybe()
    var_1 = validation_0.to_box()


def test_case_11():
    bytes_0 = b"E\xe31E\x03\x84\x85\xb9:\x9b\xc4\xaf\xbcY"
    none_type_0 = None
    validation_0 = module_0.Validation(bytes_0, none_type_0)
    var_0 = validation_0.to_lazy()
    validation_1 = module_0.Validation(bytes_0, bytes_0)
    var_1 = validation_1.to_either()
    var_2 = validation_1.to_maybe()
    var_3 = validation_1.to_box()


def test_case_12():
    bytes_0 = b"E\xe31E\x03\x84\x85\xb9:\x9b\xc4\xaf\xbcY"
    validation_0 = module_0.Validation(bytes_0, bytes_0)
    var_0 = validation_0.to_maybe()
    var_1 = validation_0.to_maybe()
    var_2 = validation_0.to_lazy()
    var_3 = var_1.__str__()
    var_4 = var_1.to_either()
    var_5 = var_2.__str__()
    var_6 = var_2.to_box()
    var_7 = var_4.to_box()
    var_5.to_either()


def test_case_13():
    bool_0 = False
    validation_0 = module_0.Validation(bool_0, bool_0)
    validation_0.to_try()


def test_case_14():
    tuple_0 = ()
    validation_0 = module_0.Validation(tuple_0, tuple_0)
    var_0 = validation_0.to_box()
    list_0 = [tuple_0]
    validation_1 = module_0.Validation(list_0, tuple_0)
    var_1 = validation_1.__str__()
    var_2 = validation_1.is_fail()
    var_3 = validation_1.__str__()
    var_2.to_lazy()


def test_case_15():
    dict_0 = {}
    validation_0 = module_0.Validation(dict_0, dict_0)
    var_0 = validation_0.to_either()
    var_1 = var_0.__str__()
    str_0 = "B4\nyP\\\nAkLz3`0V/]9"
    validation_1 = module_0.Validation(str_0, str_0)
    var_2 = validation_1.to_box()
    var_3 = validation_1.is_success()
    var_3.bind(str_0)


def test_case_16():
    bool_0 = False
    validation_0 = module_0.Validation(bool_0, bool_0)
    str_0 = "\n    Function for return function depended on first function argument\n    cond get list of two-item tuples,\n    first is condition_function, second is execute_function.\n    Returns this execute_function witch first condition_function return truly value.\n\n    :param condition_list: list of two-item tuples (condition_function, execute_function)\n    :type condition_list: List[(Function, Function)]\n    :returns: Returns this execute_function witch first condition_function return truly value\n    :rtype: Function\n    "
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0, str_0: str_0}
    validation_1 = module_0.Validation(validation_0, dict_0)
    var_0 = validation_1.__eq__(validation_0)
    var_1 = var_0.__eq__(dict_0)
    validation_2 = module_0.Validation(str_0, var_0)
    var_1.bind(validation_2)


def test_case_17():
    list_0 = []
    validation_0 = module_0.Validation(list_0, list_0)
    var_0 = validation_0.to_maybe()
    var_1 = var_0.to_try()
    var_2 = var_1.__str__()
    var_2.to_box()
