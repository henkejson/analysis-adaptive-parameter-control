# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.validation as module_0


def test_case_0():
    float_0 = -2691.3896
    set_0 = {float_0}
    str_0 = "l,@=n+\t1G7"
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    validation_1 = module_0.Validation(set_0, str_0)
    var_0 = validation_0.__eq__(str_0)
    var_0.bind(set_0)


def test_case_1():
    none_type_0 = None
    str_0 = "os$gx"
    validation_0 = module_0.Validation(none_type_0, str_0)
    var_0 = validation_0.__str__()
    var_1 = validation_0.to_try()
    var_0.to_maybe()


def test_case_2():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)


def test_case_3():
    bool_0 = False
    set_0 = {bool_0, bool_0}
    validation_0 = module_0.Validation(set_0, set_0)
    var_0 = validation_0.to_lazy()
    var_1 = validation_0.to_either()
    var_2 = var_0.to_box()


def test_case_4():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    validation_0.is_fail()


def test_case_5():
    none_type_0 = None
    set_0 = set()
    validation_0 = module_0.Validation(set_0, set_0)
    var_0 = validation_0.__eq__(none_type_0)
    var_1 = validation_0.to_box()
    var_2 = validation_0.to_maybe()
    validation_0.map(none_type_0)


def test_case_6():
    bytes_0 = b"\xa3"
    bool_0 = True
    validation_0 = module_0.Validation(bool_0, bool_0)
    validation_0.bind(bytes_0)


def test_case_7():
    bool_0 = True
    bool_1 = False
    set_0 = {bool_1}
    validation_0 = module_0.Validation(set_0, bool_1)
    validation_0.ap(bool_0)


def test_case_8():
    str_0 = "|(I\x0bKM\x0byyn5JMDYe"
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.to_box()
    var_1 = validation_0.to_either()
    var_2 = validation_0.to_box()
    var_2.map(var_2)


def test_case_9():
    int_0 = 1318
    validation_0 = module_0.Validation(int_0, int_0)
    validation_0.to_try()


def test_case_10():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    validation_1 = module_0.Validation(none_type_0, none_type_0)
    var_0 = validation_0.__eq__(validation_1)
    bytes_0 = b"\xef4\xd4"
    validation_1.ap(bytes_0)


def test_case_11():
    str_0 = "\n        If Maybe is empty or filterer returns False return default_value, in other case\n        return new instance of Maybe with the same value.\n\n        :param filterer:\n        :type filterer: Function(A) -> Boolean\n        :returns: copy of self when filterer returns True, in other case empty Maybe\n        :rtype: Maybe[A] | Maybe[None]\n        "
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.__eq__(str_0)
    var_1 = var_0.__str__()
    var_2 = validation_0.to_either()
    var_3 = var_2.to_box()
    var_4 = validation_0.to_maybe()
    var_3.map(var_4)


def test_case_12():
    none_type_0 = None
    set_0 = set()
    validation_0 = module_0.Validation(set_0, set_0)
    var_0 = validation_0.__str__()
    var_1 = validation_0.__eq__(none_type_0)
    var_2 = validation_0.to_box()
    var_3 = var_2.__eq__(var_1)
    var_4 = module_0.Validation(var_1, var_1)
    validation_0.map(none_type_0)


def test_case_13():
    str_0 = ""
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.__eq__(str_0)
    var_1 = validation_0.__eq__(validation_0)
    var_2 = var_0.__str__()
    var_3 = validation_0.to_either()
    var_4 = var_3.to_maybe()
    var_4.bind(var_3)


def test_case_14():
    str_0 = "\n        If Maybe is empty or filterer returns False return default_value, in other case\n        return new instance of Maybe with the same value.\n\n        :param filterer:\n        :type filterer: Function(A) -> Boolean\n        :returns: copy of self when filterer returns True, in other case empty Maybe\n        :rtype: Maybe[A] | Maybe[None]\n        "
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.__eq__(str_0)
    var_1 = validation_0.__eq__(validation_0)
    var_2 = var_0.__str__()
    var_3 = validation_0.to_either()
    var_4 = var_3.to_maybe()
    var_5 = var_4.bind(var_3)
    var_6 = var_3.to_box()
    var_7 = validation_0.to_maybe()
    var_8 = var_4.map(var_4)
    bool_0 = False
    set_0 = {bool_0, bool_0}
    validation_1 = module_0.Validation(set_0, set_0)
    validation_2 = module_0.Validation(var_1, var_6)
    var_9 = validation_2.__eq__(validation_1)
    var_9.to_lazy()
