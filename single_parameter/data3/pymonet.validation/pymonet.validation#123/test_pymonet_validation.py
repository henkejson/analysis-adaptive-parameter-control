# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.validation as module_0


def test_case_0():
    none_type_0 = None
    none_type_1 = None
    validation_0 = module_0.Validation(none_type_1, none_type_1)
    var_0 = validation_0.__eq__(none_type_0)


def test_case_1():
    list_0 = []
    none_type_0 = None
    validation_0 = module_0.Validation(list_0, list_0)
    var_0 = validation_0.__str__()
    var_0.ap(none_type_0)


def test_case_2():
    str_0 = "b@S,b\x0c^fK"
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.__str__()
    var_1 = var_0.__str__()
    var_1.ap(str_0)


def test_case_3():
    str_0 = "\n        Returns failed Validation with None as value and errors list.\n\n        :params errors: list of errors to store\n        :type value: List[E]\n        :returns: Failed Validation\n        :rtype: Validation[None, List[E]]\n        "
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0, str_0: str_0}
    validation_0 = module_0.Validation(dict_0, str_0)
    var_0 = validation_0.to_either()


def test_case_4():
    set_0 = set()
    validation_0 = module_0.Validation(set_0, set_0)
    var_0 = validation_0.to_maybe()


def test_case_5():
    complex_0 = 709 - 2163.1j
    validation_0 = module_0.Validation(complex_0, complex_0)


def test_case_6():
    bool_0 = True
    none_type_0 = None
    validation_0 = module_0.Validation(bool_0, none_type_0)
    validation_0.is_fail()


def test_case_7():
    complex_0 = 709 - 2163.1j
    validation_0 = module_0.Validation(complex_0, complex_0)
    validation_0.map(complex_0)


def test_case_8():
    complex_0 = 709 - 2163.1j
    validation_0 = module_0.Validation(complex_0, complex_0)
    none_type_0 = None
    validation_1 = module_0.Validation(complex_0, none_type_0)
    var_0 = validation_1.__eq__(validation_1)
    bool_0 = True
    validation_1.bind(bool_0)


def test_case_9():
    bytes_0 = b"D}s\xc7A\x97y"
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, bytes_0)
    var_0 = validation_0.is_success()
    validation_1 = module_0.Validation(none_type_0, bytes_0)
    var_1 = validation_1.to_try()
    validation_1.ap(validation_1)


def test_case_10():
    complex_0 = 709 - 2163.1j
    validation_0 = module_0.Validation(complex_0, complex_0)
    var_0 = validation_0.__eq__(complex_0)
    var_1 = validation_0.to_lazy()
    var_2 = var_1.bind(var_1)
    var_3 = var_1.__str__()
    var_4 = validation_0.to_box()
    var_0.to_try()


def test_case_11():
    bytes_0 = b"\xae\xc63lV\xfe\x14\xf3\xa33\xe8J\xbb\x1f"
    int_0 = -840
    complex_0 = 2589.8 + 607j
    validation_0 = module_0.Validation(complex_0, complex_0)
    var_0 = validation_0.to_lazy()
    var_1 = var_0.map(int_0)
    var_2 = var_1.ap(bytes_0)
    var_2.to_either()


def test_case_12():
    complex_0 = 709 - 2163.1j
    validation_0 = module_0.Validation(complex_0, complex_0)
    var_0 = validation_0.__eq__(complex_0)
    validation_0.to_try()


def test_case_13():
    none_type_0 = None
    none_type_1 = None
    validation_0 = module_0.Validation(none_type_1, none_type_0)
    var_0 = validation_0.to_lazy()
    var_1 = var_0.to_maybe()
    var_2 = var_1.to_box()
    var_2.to_box()


def test_case_14():
    str_0 = "\n        Returns failed Validation with None as value and errors list.\n\n        :params errors: list of errors to store\n        :type value: List[E]\n        :returns: Failed Validation\n        :rtype: Validation[None, List[E]]\n        "
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0, str_0: str_0}
    validation_0 = module_0.Validation(dict_0, str_0)
    var_0 = validation_0.__eq__(validation_0)


def test_case_15():
    bytes_0 = b"\xbc"
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, bytes_0)
    var_0 = validation_0.to_maybe()
    var_1 = var_0.bind(bytes_0)
    var_2 = var_1.bind(none_type_0)
    var_2.is_success()


def test_case_16():
    complex_0 = 709 - 2163.299813603551j
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    validation_1 = module_0.Validation(none_type_0, complex_0)
    var_0 = validation_1.__eq__(validation_0)
    var_0.bind(none_type_0)


def test_case_17():
    set_0 = set()
    validation_0 = module_0.Validation(set_0, set_0)
    var_0 = validation_0.to_either()
