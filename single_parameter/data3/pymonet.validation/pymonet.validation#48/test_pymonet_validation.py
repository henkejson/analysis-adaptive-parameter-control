# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.validation as module_0


def test_case_0():
    int_0 = 1787
    validation_0 = module_0.Validation(int_0, int_0)
    var_0 = validation_0.__eq__(validation_0)
    var_1 = module_0.Validation(var_0, var_0)


def test_case_1():
    str_0 = "++t3|@"
    none_type_0 = None
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.__eq__(str_0)
    var_1 = var_0.__str__()
    var_1.ap(none_type_0)


def test_case_2():
    list_0 = []
    validation_0 = module_0.Validation(list_0, list_0)
    var_0 = validation_0.__str__()
    var_1 = validation_0.is_success()
    var_1.to_maybe()


def test_case_3():
    bytes_0 = b"\xbb>\x18\xd2\xb4\x82"
    validation_0 = module_0.Validation(bytes_0, bytes_0)
    var_0 = validation_0.__str__()
    var_0.ap(bytes_0)


def test_case_4():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    var_0 = validation_0.to_box()
    str_0 = ""
    validation_1 = module_0.Validation(str_0, str_0)
    validation_2 = module_0.Validation(none_type_0, validation_1)
    var_1 = validation_1.to_either()
    int_0 = 334
    dict_0 = {str_0: int_0}
    none_type_1 = None
    validation_3 = module_0.Validation(none_type_1, none_type_1)
    validation_3.map(dict_0)


def test_case_5():
    str_0 = "g<M[B\x0cK`v-R"
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.to_maybe()
    var_0.is_success()


def test_case_6():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)


def test_case_7():
    bool_0 = True
    validation_0 = module_0.Validation(bool_0, bool_0)
    validation_0.is_success()


def test_case_8():
    bytes_0 = b"y\x00Y"
    validation_0 = module_0.Validation(bytes_0, bytes_0)
    validation_1 = module_0.Validation(validation_0, validation_0)
    validation_1.is_fail()


def test_case_9():
    bool_0 = True
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    validation_0.map(bool_0)


def test_case_10():
    none_type_0 = None
    str_0 = "\n        Returns True when errors list are not empty.\n\n        :returns: True for empty errors not list\n        :rtype: Boolean\n        "
    validation_0 = module_0.Validation(none_type_0, str_0)
    var_0 = validation_0.to_lazy()
    var_1 = validation_0.to_maybe()
    var_2 = var_1.to_box()
    var_3 = validation_0.to_box()
    var_4 = validation_0.is_fail()
    validation_0.bind(var_0)


def test_case_11():
    list_0 = []
    float_0 = 2174.68913
    validation_0 = module_0.Validation(float_0, float_0)
    validation_0.ap(list_0)


def test_case_12():
    bool_0 = True
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    var_0 = validation_0.to_box()
    var_1 = var_0.to_maybe()
    var_1.ap(bool_0)


def test_case_13():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    var_0 = validation_0.to_lazy()


def test_case_14():
    none_type_0 = None
    str_0 = "\n        Returns True when errors list are not empty.\n\n        :returns: True for empty errors not list\n        :rtype: Boolean\n        "
    validation_0 = module_0.Validation(none_type_0, str_0)
    var_0 = validation_0.to_lazy()
    var_1 = validation_0.to_maybe()
    var_2 = var_0.to_box()
    var_3 = validation_0.is_fail()
    var_4 = var_1.__eq__(validation_0)
    var_3.map(var_4)


def test_case_15():
    int_0 = 776
    validation_0 = module_0.Validation(int_0, int_0)
    validation_0.to_try()


def test_case_16():
    none_type_0 = None
    str_0 = "\n        Returns True when errors list are not empty.\n\n        :returns: True for empty errors not list\n        :rtype: Boolean\n        "
    validation_0 = module_0.Validation(none_type_0, str_0)
    var_0 = validation_0.to_lazy()
    var_1 = validation_0.to_maybe()
    var_2 = validation_0.to_either()
    var_3 = var_1.to_box()
    var_4 = validation_0.is_fail()
    var_5 = validation_0.__eq__(none_type_0)
    validation_0.map(validation_0)


def test_case_17():
    bool_0 = True
    set_0 = set()
    validation_0 = module_0.Validation(bool_0, set_0)
    var_0 = validation_0.to_maybe()
    var_0.to_maybe()


def test_case_18():
    none_type_0 = None
    int_0 = 1770
    validation_0 = module_0.Validation(int_0, int_0)
    validation_1 = module_0.Validation(validation_0, none_type_0)
    var_0 = validation_1.__eq__(validation_0)
    validation_0.to_maybe()
