# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.validation as module_0


def test_case_0():
    bytes_0 = b";\xe1r,~\x81"
    validation_0 = module_0.Validation(bytes_0, bytes_0)
    var_0 = validation_0.__eq__(bytes_0)
    var_1 = validation_0.__eq__(validation_0)
    validation_1 = module_0.Validation(var_0, bytes_0)
    var_2 = validation_0.to_try()


def test_case_1():
    str_0 = "_uX"
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.__eq__(str_0)


def test_case_2():
    set_0 = set()
    none_type_0 = None
    validation_0 = module_0.Validation(set_0, set_0)
    var_0 = validation_0.__str__()
    var_0.map(none_type_0)


def test_case_3():
    str_0 = "~cd3SN]K8kH"
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.__str__()


def test_case_4():
    list_0 = []
    validation_0 = module_0.Validation(list_0, list_0)
    var_0 = validation_0.is_fail()
    validation_1 = module_0.Validation(var_0, list_0)
    var_1 = validation_1.to_either()


def test_case_5():
    str_0 = "@[X"
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.to_either()


def test_case_6():
    str_0 = "\\"
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.to_maybe()


def test_case_7():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)


def test_case_8():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    validation_0.to_either()


def test_case_9():
    str_0 = "\n        If Maybe is empty return new empty MayUe, in other case\n        takes mapper function and returns new instance of Maybe\n        with result of mapper.\n\n        :param mapper: function to callNwith Maybe value\n        :type mapper: Function(A) -> B\n        :returns: Maybe[B | None]\n        "
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.is_fail()


def test_case_10():
    str_0 = "_X"
    validation_0 = module_0.Validation(str_0, str_0)
    validation_0.map(str_0)


def test_case_11():
    float_0 = -619.28109
    validation_0 = module_0.Validation(float_0, float_0)
    validation_0.bind(float_0)


def test_case_12():
    bytes_0 = b"R\xd0\x94\x87d~4\x1a\x8c\xecV\x1d(-\xa6B"
    validation_0 = module_0.Validation(bytes_0, bytes_0)
    validation_0.ap(bytes_0)


def test_case_13():
    str_0 = "@(xX"
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.to_box()


def test_case_14():
    str_0 = "U"
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.to_lazy()


def test_case_15():
    float_0 = -2213.0
    validation_0 = module_0.Validation(float_0, float_0)
    validation_0.to_try()


def test_case_16():
    str_0 = "}mW"
    none_type_0 = None
    validation_0 = module_0.Validation(str_0, none_type_0)
    var_0 = validation_0.to_lazy()
    var_1 = var_0.to_maybe()
    var_1.ap(str_0)


def test_case_17():
    str_0 = "@|X"
    set_0 = {str_0, str_0, str_0, str_0}
    validation_0 = module_0.Validation(str_0, set_0)
    var_0 = validation_0.__eq__(validation_0)
    var_1 = validation_0.to_maybe()
    validation_1 = module_0.Validation(set_0, var_0)
    var_2 = validation_0.is_success()
    var_3 = validation_1.__eq__(validation_0)
    var_2.to_maybe()


def test_case_18():
    bytes_0 = b""
    validation_0 = module_0.Validation(bytes_0, bytes_0)
    var_0 = validation_0.__eq__(validation_0)
    var_1 = validation_0.to_maybe()
    var_2 = var_1.__str__()
    var_2.to_try()
