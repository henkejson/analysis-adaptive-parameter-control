# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.validation as module_0


def test_case_0():
    int_0 = 706
    validation_0 = module_0.Validation(int_0, int_0)
    var_0 = validation_0.__eq__(int_0)
    var_1 = var_0.__eq__(int_0)
    var_1.is_fail()


def test_case_1():
    list_0 = []
    none_type_0 = None
    set_0 = set()
    validation_0 = module_0.Validation(none_type_0, set_0)
    var_0 = validation_0.__str__()
    var_1 = var_0.__eq__(list_0)
    var_1.to_try()


def test_case_2():
    str_0 = "\n        Return resolved Task with stored value argument.\n\n        :param value: value to store in Task\n        :type value: A\n        :returns: resolved Task\n        :rtype: Task[Function(_, resolve) -> A]\n        "
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, str_0)
    var_0 = validation_0.__str__()
    var_0.is_fail()


def test_case_3():
    set_0 = set()
    bytes_0 = b"o?\xa0\xaa-\n\xc9\xa6\xb1"
    validation_0 = module_0.Validation(bytes_0, set_0)
    var_0 = validation_0.to_either()
    var_0.to_either()


def test_case_4():
    tuple_0 = ()
    validation_0 = module_0.Validation(tuple_0, tuple_0)
    var_0 = validation_0.to_maybe()


def test_case_5():
    float_0 = -2287.07073
    validation_0 = module_0.Validation(float_0, float_0)


def test_case_6():
    int_0 = 746
    set_0 = set()
    validation_0 = module_0.Validation(set_0, set_0)
    var_0 = validation_0.is_fail()
    var_0.ap(int_0)


def test_case_7():
    bool_0 = False
    complex_0 = -1429.48 - 2329.067j
    dict_0 = {
        complex_0: complex_0,
        complex_0: complex_0,
        complex_0: complex_0,
        complex_0: complex_0,
    }
    tuple_0 = (complex_0, dict_0)
    validation_0 = module_0.Validation(tuple_0, tuple_0)
    validation_0.map(bool_0)


def test_case_8():
    float_0 = -3476.9807
    set_0 = set()
    validation_0 = module_0.Validation(set_0, set_0)
    validation_0.bind(float_0)


def test_case_9():
    float_0 = -1762.73
    bool_0 = True
    validation_0 = module_0.Validation(bool_0, bool_0)
    validation_0.ap(float_0)


def test_case_10():
    str_0 = "cc:rTfCSk,\\"
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    var_0 = validation_0.to_box()
    var_1 = var_0.__eq__(str_0)
    var_1.is_success()


def test_case_11():
    dict_0 = {}
    tuple_0 = (dict_0, dict_0)
    validation_0 = module_0.Validation(tuple_0, tuple_0)
    var_0 = validation_0.to_lazy()
    var_0.is_fail()


def test_case_12():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    var_0 = validation_0.__eq__(none_type_0)
    var_1 = validation_0.to_lazy()
    var_2 = var_1.bind(var_0)
    var_3 = var_1.to_box()
    var_1.is_fail()


def test_case_13():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    validation_0.to_try()


def test_case_14():
    tuple_0 = ()
    validation_0 = module_0.Validation(tuple_0, tuple_0)
    var_0 = validation_0.__eq__(validation_0)
    tuple_0.map(tuple_0)


def test_case_15():
    str_0 = "cc:rTfCSk,\\"
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.to_maybe()


def test_case_16():
    complex_0 = -1429.48 - 2329.067j
    dict_0 = {
        complex_0: complex_0,
        complex_0: complex_0,
        complex_0: complex_0,
        complex_0: complex_0,
    }
    tuple_0 = (complex_0, dict_0)
    validation_0 = module_0.Validation(tuple_0, tuple_0)
    var_0 = validation_0.to_lazy()
    var_1 = validation_0.to_either()
    var_0.is_success()


def test_case_17():
    tuple_0 = ()
    validation_0 = module_0.Validation(tuple_0, tuple_0)
    var_0 = validation_0.__eq__(validation_0)
    validation_1 = module_0.Validation(var_0, tuple_0)
    complex_0 = -396.844732 - 1881.008j
    validation_2 = module_0.Validation(complex_0, var_0)
    var_1 = validation_2.__eq__(validation_1)
    var_1.map(validation_1)
