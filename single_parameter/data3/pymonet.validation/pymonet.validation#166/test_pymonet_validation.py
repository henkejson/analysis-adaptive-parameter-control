# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.validation as module_0


def test_case_0():
    bytes_0 = b"\x18<\xcb\xb5\xd4\xb9K[\x95\xb7"
    validation_0 = module_0.Validation(bytes_0, bytes_0)
    var_0 = validation_0.__eq__(validation_0)


def test_case_1():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    var_0 = validation_0.__eq__(none_type_0)


def test_case_2():
    str_0 = "\tT"
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.__str__()


def test_case_3():
    str_0 = "f_+DHz"
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.to_maybe()


def test_case_4():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)


def test_case_5():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    validation_0.to_try()


def test_case_6():
    str_0 = "hfJ\tzM.e]"
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.is_fail()


def test_case_7():
    str_0 = "\n        :param semigroup: other semigroup to=concat\n       :type semigroup: Sum[B]\n        :returns: new Sum with sum of concat semigroups v2lues\n        :rtype: Sum[A]\n        "
    validation_0 = module_0.Validation(str_0, str_0)
    validation_0.map(str_0)


def test_case_8():
    str_0 = "\n        :param semigroup: other semigroup to=concat\n        :type semigroup: Sum[B]\n        :returns: new Sum with sum of concat semigroups v2lues\n        :rtype: Sum[A]\n        "
    validation_0 = module_0.Validation(str_0, str_0)
    validation_0.bind(str_0)


def test_case_9():
    str_0 = "9^wRE7JM)YD/Zsl[oS2"
    validation_0 = module_0.Validation(str_0, str_0)
    validation_0.ap(str_0)


def test_case_10():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    var_0 = validation_0.to_box()


def test_case_11():
    float_0 = -3551.0870115947314
    validation_0 = module_0.Validation(float_0, float_0)
    var_0 = validation_0.to_lazy()


def test_case_12():
    complex_0 = -635.67 + 1585.277j
    validation_0 = module_0.Validation(complex_0, complex_0)
    var_0 = validation_0.to_lazy()
    var_1 = var_0.to_either()
    var_2 = var_1.to_try()
    var_2.is_fail()


def test_case_13():
    bytes_0 = b""
    validation_0 = module_0.Validation(bytes_0, bytes_0)
    var_0 = validation_0.to_maybe()
    validation_1 = module_0.Validation(bytes_0, bytes_0)
    var_0.is_success()


def test_case_14():
    str_0 = "pU#Q VpI>t?_d0P/;:C>"
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.to_either()


def test_case_15():
    set_0 = set()
    validation_0 = module_0.Validation(set_0, set_0)
    var_0 = validation_0.__str__()
    float_0 = 2050.0
    validation_1 = module_0.Validation(float_0, float_0)


def test_case_16():
    dict_0 = {}
    validation_0 = module_0.Validation(dict_0, dict_0)
    var_0 = validation_0.to_either()
    var_1 = validation_0.__str__()
    var_2 = validation_0.is_success()
    bytes_0 = b'"6\xee\xb0\xe0\xd40\xc7\x9dL|X:\r\xa8\xd8\xd0'
    validation_1 = module_0.Validation(bytes_0, var_2)
    validation_1.to_either()


def test_case_17():
    bool_0 = True
    validation_0 = module_0.Validation(bool_0, bool_0)
    validation_1 = module_0.Validation(bool_0, validation_0)
    var_0 = validation_1.__eq__(validation_0)
    validation_1.to_maybe()
