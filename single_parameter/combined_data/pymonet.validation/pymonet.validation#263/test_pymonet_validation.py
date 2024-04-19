# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.validation as module_0
import builtins as module_1


def test_case_0():
    int_0 = 1
    validation_0 = module_0.Validation(int_0, int_0)
    var_0 = validation_0.__eq__(validation_0)
    var_0.to_try()


def test_case_1():
    bool_0 = False
    none_type_0 = None
    none_type_1 = None
    validation_0 = module_0.Validation(bool_0, none_type_0)
    var_0 = validation_0.__eq__(none_type_1)
    var_0.to_try()


def test_case_2():
    str_0 = "Maybe[U]"
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.__str__()
    var_0.is_success()


def test_case_3():
    bytes_0 = b" Q(Z"
    validation_0 = module_0.Validation(bytes_0, bytes_0)


def test_case_4():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    validation_0.is_success()


def test_case_5():
    dict_0 = {}
    object_0 = module_1.object(**dict_0)
    str_0 = "\n        Return resolved Task with stored value argument.\n\n        :param value: value to store in Task\n        :type value: A\n        :returns: resolved Task\n        :rtype: Task[Function(_, resolve) -> A]\n        "
    validation_0 = module_0.Validation(str_0, str_0)
    int_0 = 2406
    validation_1 = module_0.Validation(int_0, int_0)
    validation_1.is_fail()


def test_case_6():
    str_0 = "I"
    str_1 = "\x0b"
    validation_0 = module_0.Validation(str_1, str_1)
    validation_0.map(str_0)


def test_case_7():
    int_0 = 1
    validation_0 = module_0.Validation(int_0, int_0)
    var_0 = validation_0.__eq__(validation_0)
    validation_0.bind(int_0)


def test_case_8():
    float_0 = 697.939649
    int_0 = 2196
    validation_0 = module_0.Validation(int_0, int_0)
    validation_0.ap(float_0)


def test_case_9():
    str_0 = "!g\x0c:p @h;Is"
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.to_box()
    var_0.is_fail()


def test_case_10():
    bool_0 = True
    str_0 = "{pjJ) jF%K"
    set_0 = {bool_0, bool_0, bool_0, str_0}
    validation_0 = module_0.Validation(set_0, str_0)
    none_type_0 = None
    var_0 = validation_0.to_lazy()
    validation_0.map(none_type_0)


def test_case_11():
    str_0 = "}^"
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.to_either()
    var_1 = var_0.to_lazy()
    var_2 = var_1.to_either()
    complex_0 = 327.1 + 4324.044879j
    validation_1 = module_0.Validation(complex_0, complex_0)
    var_3 = validation_1.to_lazy()
    var_4 = validation_1.to_box()
    var_5 = var_3.to_box()
    var_6 = validation_0.is_fail()
    var_7 = var_3.__str__()
    var_4.bind(str_0)


def test_case_12():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    validation_0.to_try()


def test_case_13():
    tuple_0 = ()
    none_type_0 = None
    validation_0 = module_0.Validation(tuple_0, tuple_0)
    var_0 = validation_0.is_fail()
    var_1 = validation_0.__eq__(tuple_0)
    var_2 = validation_0.__str__()
    var_3 = validation_0.__eq__(none_type_0)
    var_4 = validation_0.to_try()
    var_0.map(var_0)


def test_case_14():
    bool_0 = True
    bytes_0 = b"TC\xcf*\x86A\xcc9\xa5#SY"
    validation_0 = module_0.Validation(bytes_0, bytes_0)
    var_0 = validation_0.to_maybe()
    var_1 = var_0.bind(bool_0)


def test_case_15():
    int_0 = 1
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, int_0)
    validation_1 = module_0.Validation(int_0, int_0)
    var_0 = validation_1.__eq__(int_0)
    validation_2 = module_0.Validation(validation_1, var_0)
    var_1 = validation_2.to_lazy()
    var_2 = validation_2.__eq__(validation_1)
    var_3 = var_0.__str__()
    var_4 = var_1.__eq__(var_1)
    var_0.bind(validation_2)


def test_case_16():
    float_0 = 2357.72
    bytes_0 = b"\x12\xaf\x8f\x0fy9&"
    validation_0 = module_0.Validation(float_0, bytes_0)
    var_0 = validation_0.to_either()
    var_1 = var_0.__str__()
    var_1.to_maybe()


def test_case_17():
    str_0 = ""
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.to_either()
    var_1 = var_0.to_lazy()
    var_0.is_success()


def test_case_18():
    str_0 = ""
    validation_0 = module_0.Validation(str_0, str_0)
    validation_1 = module_0.Validation(str_0, validation_0)
    var_0 = validation_0.to_maybe()
    var_1 = validation_0.to_either()
    var_2 = var_1.to_lazy()
    var_3 = var_2.to_either()
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0, str_0: str_0}
    var_4 = var_0.to_lazy()
    var_5 = validation_1.to_box()
    var_6 = var_5.__eq__(dict_0)
    var_7 = validation_0.is_fail()
    var_8 = var_6.__eq__(var_7)
    var_9 = var_4.bind(var_5)
    var_2.is_fail()
