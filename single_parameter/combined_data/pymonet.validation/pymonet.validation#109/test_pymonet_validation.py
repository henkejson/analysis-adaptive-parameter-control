# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.validation as module_0


def test_case_0():
    bool_0 = True
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    var_0 = validation_0.__eq__(bool_0)
    var_0.is_success()


def test_case_1():
    bool_0 = True
    tuple_0 = ()
    bytes_0 = b"+\xa3\xa1\r:\x90\x9cF\xde\xd9M\x0b\xf6BN[\xce\xe6l\xe5"
    dict_0 = {tuple_0: bool_0, bytes_0: bool_0, bool_0: bool_0}
    validation_0 = module_0.Validation(bool_0, dict_0)
    var_0 = validation_0.__str__()
    var_0.to_box()


def test_case_2():
    dict_0 = {}
    validation_0 = module_0.Validation(dict_0, dict_0)
    var_0 = validation_0.to_either()
    var_0.is_success()


def test_case_3():
    bytes_0 = b"\xd5\x83\x03"
    validation_0 = module_0.Validation(bytes_0, bytes_0)
    var_0 = validation_0.to_either()
    validation_1 = module_0.Validation(validation_0, bytes_0)
    var_1 = var_0.to_try()
    var_2 = validation_1.is_success()
    var_3 = var_0.map(var_1)
    var_4 = validation_1.__eq__(var_1)
    var_1.to_maybe()


def test_case_4():
    set_0 = set()
    validation_0 = module_0.Validation(set_0, set_0)
    var_0 = validation_0.to_maybe()
    var_1 = var_0.__str__()
    var_2 = var_1.__eq__(validation_0)
    var_1.to_lazy()


def test_case_5():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)


def test_case_6():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    validation_0.to_try()


def test_case_7():
    dict_0 = {}
    validation_0 = module_0.Validation(dict_0, dict_0)
    int_0 = -566
    none_type_0 = None
    validation_1 = module_0.Validation(int_0, none_type_0)
    validation_1.is_fail()


def test_case_8():
    bytes_0 = b"%"
    none_type_0 = None
    validation_0 = module_0.Validation(bytes_0, none_type_0)
    var_0 = validation_0.to_lazy()
    validation_0.map(bytes_0)


def test_case_9():
    bool_0 = False
    str_0 = "G/"
    validation_0 = module_0.Validation(str_0, str_0)
    validation_0.bind(bool_0)


def test_case_10():
    bool_0 = False
    set_0 = {bool_0}
    bytes_0 = b"\xa24\xa0\x86\xbd+\xf2\x19\xe1\x89"
    validation_0 = module_0.Validation(bytes_0, bytes_0)
    validation_0.ap(set_0)


def test_case_11():
    bool_0 = True
    dict_0 = {bool_0: bool_0, bool_0: bool_0, bool_0: bool_0}
    validation_0 = module_0.Validation(bool_0, dict_0)
    var_0 = validation_0.to_lazy()
    var_1 = var_0.to_either()
    var_2 = var_1.to_box()
    var_3 = var_2.to_lazy()
    var_4 = var_3.to_either()
    dict_1 = {}
    validation_1 = module_0.Validation(bool_0, dict_1)
    validation_2 = module_0.Validation(bool_0, bool_0)
    var_5 = validation_2.to_box()
    validation_2.bind(bool_0)


def test_case_12():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    var_0 = validation_0.to_lazy()
    var_1 = var_0.to_either()
    var_2 = var_0.ap(none_type_0)
    var_3 = var_0.to_either()
    var_4 = var_3.__str__()
    var_4.to_try()


def test_case_13():
    bool_0 = True
    dict_0 = {bool_0: bool_0, bool_0: bool_0, bool_0: bool_0}
    validation_0 = module_0.Validation(bool_0, dict_0)
    var_0 = validation_0.__str__()
    validation_1 = module_0.Validation(bool_0, bool_0)
    var_1 = validation_0.is_fail()
    var_2 = validation_0.to_maybe()
    var_3 = validation_1.__eq__(bool_0)
    validation_1.is_success()


def test_case_14():
    dict_0 = {}
    validation_0 = module_0.Validation(dict_0, dict_0)
    none_type_0 = None
    var_0 = validation_0.__eq__(validation_0)
    var_1 = var_0.__eq__(none_type_0)
    var_2 = validation_0.__eq__(validation_0)
    var_3 = var_2.__eq__(dict_0)


def test_case_15():
    bool_0 = True
    dict_0 = {bool_0: bool_0, bool_0: bool_0, bool_0: bool_0}
    validation_0 = module_0.Validation(bool_0, dict_0)
    var_0 = validation_0.__str__()
    validation_1 = module_0.Validation(bool_0, bool_0)
    var_1 = validation_0.is_fail()
    var_2 = validation_1.__eq__(bool_0)
    var_3 = validation_1.__eq__(validation_0)
    validation_1.ap(bool_0)


def test_case_16():
    dict_0 = {}
    validation_0 = module_0.Validation(dict_0, dict_0)
    int_0 = -566
    none_type_0 = None
    var_0 = validation_0.__str__()
    validation_1 = module_0.Validation(int_0, none_type_0)
    validation_1.to_maybe()
