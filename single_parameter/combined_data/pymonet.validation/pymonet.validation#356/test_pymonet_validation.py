# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.validation as module_0


def test_case_0():
    bool_0 = True
    validation_0 = module_0.Validation(bool_0, bool_0)
    var_0 = validation_0.__eq__(validation_0)


def test_case_1():
    set_0 = set()
    validation_0 = module_0.Validation(set_0, set_0)
    var_0 = validation_0.__eq__(validation_0)
    bool_0 = True
    bool_1 = False
    var_1 = validation_0.__eq__(bool_0)
    validation_1 = module_0.Validation(validation_0, bool_1)
    var_2 = var_0.__eq__(var_1)
    var_1.is_fail()


def test_case_2():
    list_0 = []
    validation_0 = module_0.Validation(list_0, list_0)
    var_0 = validation_0.__str__()
    var_0.to_either()


def test_case_3():
    bool_0 = False
    int_0 = -2149
    list_0 = [bool_0, int_0, int_0]
    validation_0 = module_0.Validation(bool_0, list_0)
    var_0 = validation_0.__str__()
    var_0.to_either()


def test_case_4():
    bytes_0 = b"\r\xd9\x9fy\xe1\xb4@~\x94eC&\xaf\xfe\xc9\x8aV"
    validation_0 = module_0.Validation(bytes_0, bytes_0)
    var_0 = validation_0.to_either()
    var_1 = var_0.to_maybe()
    var_2 = var_1.ap(var_0)
    var_3 = var_1.to_try()


def test_case_5():
    set_0 = set()
    validation_0 = module_0.Validation(set_0, set_0)
    var_0 = validation_0.to_lazy()
    var_1 = var_0.to_box()
    var_2 = validation_0.to_maybe()
    var_3 = var_2.__str__()
    var_0.is_success()


def test_case_6():
    none_type_0 = None
    bool_0 = True
    dict_0 = {none_type_0: none_type_0, bool_0: none_type_0}
    validation_0 = module_0.Validation(none_type_0, dict_0)
    var_0 = validation_0.to_maybe()
    var_1 = var_0.ap(none_type_0)
    var_1.is_success()


def test_case_7():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)


def test_case_8():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    validation_0.__str__()


def test_case_9():
    bool_0 = True
    validation_0 = module_0.Validation(bool_0, bool_0)
    validation_0.is_fail()


def test_case_10():
    none_type_0 = None
    str_0 = 'UOU*"%QZ&5s<-Y'
    validation_0 = module_0.Validation(str_0, str_0)
    validation_0.map(none_type_0)


def test_case_11():
    set_0 = set()
    validation_0 = module_0.Validation(set_0, set_0)
    var_0 = validation_0.__eq__(validation_0)
    validation_0.bind(set_0)


def test_case_12():
    bool_0 = False
    bytes_0 = b"\xb1\xa8f>(\xf1:\xd6P\xbd"
    validation_0 = module_0.Validation(bytes_0, bytes_0)
    validation_0.ap(bool_0)


def test_case_13():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    var_0 = validation_0.to_box()
    var_0.is_success()


def test_case_14():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    validation_0.to_try()


def test_case_15():
    set_0 = set()
    validation_0 = module_0.Validation(set_0, set_0)
    var_0 = validation_0.to_either()
    var_1 = validation_0.to_lazy()
    var_2 = validation_0.to_try()
    var_3 = validation_0.to_maybe()
    var_4 = validation_0.is_fail()
    var_4.ap(var_0)


def test_case_16():
    set_0 = set()
    validation_0 = module_0.Validation(set_0, set_0)
    var_0 = validation_0.__eq__(validation_0)
    bool_0 = True
    var_1 = var_0.__eq__(validation_0)
    var_2 = validation_0.to_try()
    none_type_0 = None
    validation_1 = module_0.Validation(none_type_0, bool_0)
    var_3 = validation_1.__eq__(validation_0)
    validation_1.is_fail()
