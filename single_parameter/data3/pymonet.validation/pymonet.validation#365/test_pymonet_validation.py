# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.validation as module_0


def test_case_0():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    var_0 = validation_0.__eq__(validation_0)
    var_0.is_success()


def test_case_1():
    bool_0 = True
    tuple_0 = (bool_0,)
    validation_0 = module_0.Validation(bool_0, tuple_0)
    var_0 = validation_0.__eq__(tuple_0)
    var_1 = validation_0.to_maybe()


def test_case_2():
    list_0 = []
    validation_0 = module_0.Validation(list_0, list_0)
    var_0 = validation_0.__str__()
    var_0.to_try()


def test_case_3():
    none_type_0 = None
    none_type_0.is_success()


def test_case_4():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)


def test_case_5():
    int_0 = -1619
    validation_0 = module_0.Validation(int_0, int_0)
    validation_0.to_either()


def test_case_6():
    bool_0 = False
    validation_0 = module_0.Validation(bool_0, bool_0)
    validation_0.is_fail()


def test_case_7():
    none_type_0 = None
    bool_0 = True
    tuple_0 = (bool_0,)
    validation_0 = module_0.Validation(bool_0, tuple_0)
    var_0 = validation_0.__eq__(tuple_0)
    var_1 = validation_0.to_maybe()
    var_2 = validation_0.__eq__(none_type_0)
    list_0 = [var_0, var_2, validation_0, validation_0]
    validation_0.map(list_0)


def test_case_8():
    bytes_0 = b"\x91\xe4\xc1\x1d!\xa4\x12\xd4]K"
    int_0 = 1587
    validation_0 = module_0.Validation(int_0, int_0)
    validation_0.bind(bytes_0)


def test_case_9():
    int_0 = 1023
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    validation_0.ap(int_0)


def test_case_10():
    int_0 = 1587
    validation_0 = module_0.Validation(int_0, int_0)
    var_0 = validation_0.to_box()


def test_case_11():
    int_0 = 1587
    validation_0 = module_0.Validation(int_0, int_0)
    var_0 = validation_0.to_lazy()


def test_case_12():
    int_0 = -625
    str_0 = "dx\x0b|Wrn~w?c:pCxu\x0c"
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.to_lazy()
    var_1 = var_0.to_maybe()
    var_2 = var_1.__str__()
    var_2.bind(int_0)


def test_case_13():
    int_0 = -1619
    validation_0 = module_0.Validation(int_0, int_0)
    validation_0.to_try()


def test_case_14():
    str_0 = "dx\x0b|Wrn~w?c:pCxu\x0c"
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.to_either()
    var_1 = validation_0.__str__()
    var_1.ap(validation_0)


def test_case_15():
    list_0 = []
    validation_0 = module_0.Validation(list_0, list_0)
    int_0 = 544
    var_0 = validation_0.__eq__(int_0)
    none_type_0 = None
    var_1 = validation_0.__eq__(none_type_0)
    var_2 = validation_0.to_either()
    var_1.is_success()


def test_case_16():
    none_type_0 = None
    bool_0 = True
    bool_1 = True
    tuple_0 = (bool_1,)
    validation_0 = module_0.Validation(bool_0, tuple_0)
    var_0 = validation_0.__eq__(tuple_0)
    validation_1 = module_0.Validation(none_type_0, none_type_0)
    var_1 = validation_1.__eq__(validation_0)
    var_2 = validation_0.to_maybe()
    var_3 = var_2.to_either()


def test_case_17():
    bool_0 = True
    list_0 = []
    validation_0 = module_0.Validation(list_0, list_0)
    var_0 = validation_0.to_maybe()
    var_0.ap(bool_0)
