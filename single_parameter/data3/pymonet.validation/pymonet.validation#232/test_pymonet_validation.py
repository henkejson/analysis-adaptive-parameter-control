# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.validation as module_0


def test_case_0():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    validation_1 = module_0.Validation(none_type_0, none_type_0)
    var_0 = validation_1.__eq__(validation_0)
    validation_0.to_maybe()


def test_case_1():
    none_type_0 = None
    none_type_1 = None
    validation_0 = module_0.Validation(none_type_0, none_type_1)
    var_0 = validation_0.__eq__(none_type_1)
    var_0.to_maybe()


def test_case_2():
    str_0 = "Re^kL;WE"
    set_0 = set()
    validation_0 = module_0.Validation(set_0, set_0)
    var_0 = validation_0.__str__()
    list_0 = [var_0, var_0]
    validation_1 = module_0.Validation(list_0, validation_0)
    validation_1.bind(str_0)


def test_case_3():
    str_0 = "\n        Take function and applied this function on current Validation value and returns folder result.\n\n        :param mapper: mapper function\n        :type mapper: Function(A) -> Validation[B, E]\n        :returns: new Validation with mapped value\n        :rtype: Validation[B, E]\n        "
    complex_0 = 457.16 - 2025.286325j
    str_1 = " U<zRK0mA8`D\x0c.Oa-"
    validation_0 = module_0.Validation(str_1, str_1)
    var_0 = validation_0.is_fail()
    validation_1 = module_0.Validation(complex_0, str_0)
    var_1 = validation_1.__str__()
    var_1.to_either()


def test_case_4():
    tuple_0 = ()
    validation_0 = module_0.Validation(tuple_0, tuple_0)


def test_case_5():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    validation_0.to_maybe()


def test_case_6():
    bool_0 = False
    validation_0 = module_0.Validation(bool_0, bool_0)
    validation_0.is_fail()


def test_case_7():
    int_0 = 3325
    list_0 = [int_0, int_0, int_0]
    tuple_0 = ()
    validation_0 = module_0.Validation(tuple_0, tuple_0)
    validation_0.map(list_0)


def test_case_8():
    tuple_0 = ()
    none_type_0 = None
    validation_0 = module_0.Validation(tuple_0, none_type_0)
    validation_0.bind(validation_0)


def test_case_9():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    validation_1 = module_0.Validation(none_type_0, none_type_0)
    var_0 = validation_1.__eq__(validation_0)
    validation_1.ap(validation_0)


def test_case_10():
    int_0 = 2995
    validation_0 = module_0.Validation(int_0, int_0)
    var_0 = validation_0.to_box()
    var_0.is_success()


def test_case_11():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    var_0 = validation_0.to_lazy()
    var_1 = var_0.__str__()
    var_1.to_lazy()


def test_case_12():
    dict_0 = {}
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, dict_0)
    validation_1 = module_0.Validation(dict_0, dict_0)
    var_0 = validation_1.to_lazy()
    var_1 = var_0.to_either()
    validation_1.map(validation_1)


def test_case_13():
    bool_0 = False
    validation_0 = module_0.Validation(bool_0, bool_0)
    validation_0.to_try()


def test_case_14():
    bytes_0 = b"WV\x16\xb9"
    validation_0 = module_0.Validation(bytes_0, bytes_0)
    var_0 = validation_0.to_either()
    var_1 = var_0.ap(bytes_0)
    var_2 = var_1.__str__()
    var_2.to_try()


def test_case_15():
    dict_0 = {}
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, dict_0)
    validation_1 = validation_0.to_maybe()
    var_0 = validation_1.to_lazy()
    validation_1.to_maybe()


def test_case_16():
    none_type_0 = None
    str_0 = "uE4P"
    validation_0 = module_0.Validation(str_0, str_0)
    validation_1 = module_0.Validation(none_type_0, validation_0)
    validation_2 = validation_0.to_maybe()
    var_0 = validation_1.__eq__(str_0)
    var_1 = validation_2.to_either()
    validation_2.to_maybe()


def test_case_17():
    none_type_0 = None
    str_0 = "uiP"
    validation_0 = module_0.Validation(str_0, str_0)
    validation_1 = module_0.Validation(str_0, validation_0)
    validation_2 = module_0.Validation(str_0, none_type_0)
    var_0 = validation_0.__eq__(none_type_0)
    var_1 = validation_2.__eq__(validation_1)
    var_1.to_either()


def test_case_18():
    none_type_0 = None
    str_0 = ""
    validation_0 = module_0.Validation(str_0, str_0)
    validation_1 = module_0.Validation(str_0, validation_0)
    var_0 = validation_0.__eq__(none_type_0)
    var_1 = validation_0.to_either()
    var_2 = var_1.__eq__(validation_1)
    var_3 = validation_0.is_fail()
    var_3.to_box()
