# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.validation as module_0


def test_case_0():
    bool_0 = True
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, bool_0)
    var_0 = validation_0.__eq__(bool_0)
    validation_0.is_fail()


def test_case_1():
    int_0 = 1175
    dict_0 = {}
    tuple_0 = (int_0, dict_0)
    validation_0 = module_0.Validation(int_0, tuple_0)
    var_0 = validation_0.__str__()
    var_0.to_try()


def test_case_2():
    int_0 = 1175
    dict_0 = {}
    tuple_0 = (int_0, dict_0)
    validation_0 = module_0.Validation(int_0, tuple_0)
    var_0 = validation_0.__str__()
    var_1 = validation_0.to_either()
    var_0.to_try()


def test_case_3():
    int_0 = 1175
    dict_0 = {}
    tuple_0 = (int_0, dict_0)
    validation_0 = module_0.Validation(int_0, tuple_0)
    var_0 = validation_0.to_maybe()
    var_1 = validation_0.__eq__(validation_0)
    validation_0.ap(var_1)


def test_case_4():
    int_0 = -1015
    validation_0 = module_0.Validation(int_0, int_0)


def test_case_5():
    int_0 = 0
    validation_0 = module_0.Validation(int_0, int_0)
    validation_0.to_maybe()


def test_case_6():
    none_type_0 = None
    dict_0 = {none_type_0: none_type_0}
    validation_0 = module_0.Validation(none_type_0, dict_0)
    var_0 = validation_0.to_lazy()
    var_1 = var_0.to_either()
    var_2 = var_1.to_lazy()
    str_0 = "\n        Take mapper function and return new instance of Right with mapped value.\n\n        :param mapper: function to apply on Right value\n        :type mapper: Function(A) -> B\n        :returns: new Right with result of mapper\n        :rtype: Right[B]\n        "
    validation_1 = module_0.Validation(str_0, str_0)
    validation_1.map(none_type_0)


def test_case_7():
    bool_0 = False
    bool_1 = False
    validation_0 = module_0.Validation(bool_1, bool_1)
    validation_0.bind(bool_0)


def test_case_8():
    bool_0 = False
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    validation_0.ap(bool_0)


def test_case_9():
    float_0 = -1381.0787
    validation_0 = module_0.Validation(float_0, float_0)
    var_0 = validation_0.to_box()
    validation_0.is_success()


def test_case_10():
    bytes_0 = b"\x11\xf6\xa0V\xb6\x915:\x86R$\x90.m"
    none_type_0 = None
    validation_0 = module_0.Validation(bytes_0, bytes_0)
    var_0 = validation_0.to_lazy()
    var_1 = var_0.to_maybe()
    var_2 = var_1.__str__()
    var_3 = var_2.__str__()
    var_3.map(none_type_0)


def test_case_11():
    bytes_0 = b"8\x83\xdf\x07\xb4\x8f\xb2\x0f\xa5\xe0"
    set_0 = {bytes_0}
    none_type_0 = None
    validation_0 = module_0.Validation(set_0, none_type_0)
    validation_0.to_try()


def test_case_12():
    list_0 = []
    list_1 = []
    validation_0 = module_0.Validation(list_1, list_0)
    var_0 = validation_0.to_either()
    var_1 = validation_0.__eq__(list_1)


def test_case_13():
    int_0 = 1175
    tuple_0 = (int_0, int_0)
    validation_0 = module_0.Validation(int_0, tuple_0)
    var_0 = validation_0.__eq__(validation_0)
    var_0.to_try()


def test_case_14():
    bytes_0 = b">\x96\x14q\x01X"
    validation_0 = module_0.Validation(bytes_0, bytes_0)
    validation_1 = module_0.Validation(bytes_0, validation_0)
    var_0 = validation_1.__eq__(validation_0)
    validation_0.map(bytes_0)


def test_case_15():
    int_0 = 1175
    dict_0 = {}
    validation_0 = module_0.Validation(int_0, dict_0)
    var_0 = validation_0.to_maybe()
    var_0.to_maybe()


def test_case_16():
    dict_0 = {}
    validation_0 = module_0.Validation(dict_0, dict_0)
    var_0 = validation_0.__str__()
    var_0.to_try()
