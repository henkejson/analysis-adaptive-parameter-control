# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.validation as module_0


def test_case_0():
    bool_0 = False
    set_0 = {bool_0, bool_0}
    validation_0 = module_0.Validation(set_0, set_0)
    var_0 = validation_0.__str__()
    var_1 = validation_0.to_try()
    none_type_0 = None
    var_2 = var_1.__eq__(none_type_0)
    var_1.to_try()


def test_case_1():
    dict_0 = {}
    validation_0 = module_0.Validation(dict_0, dict_0)
    var_0 = validation_0.to_either()
    var_1 = var_0.__eq__(dict_0)
    var_1.to_either()


def test_case_2():
    tuple_0 = ()
    tuple_1 = ()
    validation_0 = module_0.Validation(tuple_1, tuple_0)
    var_0 = validation_0.to_maybe()
    var_0.is_success()


def test_case_3():
    str_0 = "\n        Transform Lazy into Try with constructor_fn result.\n        Try will be successful only when constructor_fn not raise anything.\n\n        :returns: Try with constructor_fn result\n        :rtype: Try[A] | Try[Error]\n        "
    set_0 = {str_0, str_0}
    validation_0 = module_0.Validation(set_0, set_0)
    var_0 = validation_0.to_maybe()
    var_1 = validation_0.to_box()
    var_2 = validation_0.__str__()


def test_case_4():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)


def test_case_5():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    validation_0.to_either()


def test_case_6():
    bool_0 = True
    validation_0 = module_0.Validation(bool_0, bool_0)
    validation_0.is_fail()


def test_case_7():
    int_0 = 15
    validation_0 = module_0.Validation(int_0, int_0)
    var_0 = validation_0.__eq__(validation_0)
    validation_0.map(validation_0)


def test_case_8():
    int_0 = 15
    validation_0 = module_0.Validation(int_0, int_0)
    var_0 = validation_0.__eq__(validation_0)
    validation_0.bind(int_0)


def test_case_9():
    str_0 = "\n        Transform Lazy into Try with constructor_fn result.\n        Try wi[l be successful only when constructor_fn not raise anything.\n\n q      :returns: Try with constructor_fn result\n        :rtype: Try[A] | Try[Error]\n        "
    set_0 = {str_0, str_0}
    validation_0 = module_0.Validation(set_0, set_0)
    var_0 = validation_0.__eq__(validation_0)
    var_1 = validation_0.is_fail()
    complex_0 = 21 + 1084.99j
    validation_0.ap(complex_0)


def test_case_10():
    str_0 = "\n        Transform Lazy into Try with constructor_fn result.\n        Try will be successful only when constructor_fn not raise anything.\n\n        :returns: Try with constructor_fn result\n        :rtype: Try[A] | Try[Error]\n        "
    set_0 = {str_0, str_0}
    validation_0 = module_0.Validation(set_0, set_0)
    var_0 = validation_0.to_box()
    var_1 = validation_0.__str__()


def test_case_11():
    float_0 = 753.87
    validation_0 = module_0.Validation(float_0, float_0)
    var_0 = validation_0.to_lazy()


def test_case_12():
    dict_0 = {}
    validation_0 = module_0.Validation(dict_0, dict_0)
    var_0 = validation_0.to_lazy()
    var_1 = var_0.to_box()
    var_2 = var_0.to_try()
    var_2.to_maybe()


def test_case_13():
    bool_0 = False
    set_0 = {bool_0, bool_0}
    validation_0 = module_0.Validation(set_0, set_0)
    var_0 = validation_0.to_try()
    none_type_0 = None
    var_1 = var_0.__eq__(none_type_0)
    var_0.to_try()


def test_case_14():
    str_0 = "\n        Transform Lazy into Try with constructor_fn result.\n        Try wi[l be successful only when constructor_fn not raise anything.\n\n q      :returns: Try with constructor_fn result\n        :rtype: Try[A] | Try[Error]\n        "
    set_0 = {str_0, str_0}
    validation_0 = module_0.Validation(set_0, set_0)
    var_0 = validation_0.__eq__(validation_0)
    var_1 = validation_0.is_fail()
    var_2 = validation_0.to_either()
    var_0.to_either()


def test_case_15():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    var_0 = validation_0.__eq__(none_type_0)


def test_case_16():
    set_0 = set()
    validation_0 = module_0.Validation(set_0, set_0)
    var_0 = validation_0.__eq__(set_0)
    var_1 = validation_0.is_fail()
    var_2 = validation_0.__str__()
    var_3 = validation_0.to_either()
    complex_0 = 21 + 1084.99j
    var_0.ap(complex_0)


def test_case_17():
    int_0 = 15
    validation_0 = module_0.Validation(int_0, int_0)
    var_0 = validation_0.__eq__(validation_0)


def test_case_18():
    set_0 = set()
    bytes_0 = b"\x0f&\xaf\xa9\xf4\xe4\xe8F\x1b\xc3\x1f\x07\xdb+\xa7W\x00\xc3\xb7\xa3"
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, bytes_0)
    list_0 = []
    validation_1 = module_0.Validation(list_0, set_0)
    var_0 = validation_0.to_box()
    var_1 = validation_0.__eq__(validation_1)
    var_0.map(list_0)
