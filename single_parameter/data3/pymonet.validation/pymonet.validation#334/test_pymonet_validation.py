# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.validation as module_0


def test_case_0():
    bool_0 = False
    bool_1 = False
    validation_0 = module_0.Validation(bool_0, bool_0)
    var_0 = validation_0.__eq__(bool_1)
    var_1 = validation_0.__eq__(bool_1)
    validation_0.to_either()


def test_case_1():
    str_0 = "\x0bp"
    list_0 = [str_0, str_0]
    validation_0 = module_0.Validation(str_0, list_0)
    var_0 = validation_0.__str__()
    var_0.to_maybe()


def test_case_2():
    list_0 = []
    validation_0 = module_0.Validation(list_0, list_0)
    var_0 = validation_0.__eq__(validation_0)
    var_1 = validation_0.is_success()
    var_2 = validation_0.to_either()
    var_2.to_either()


def test_case_3():
    tuple_0 = ()
    str_0 = "xyut"
    validation_0 = module_0.Validation(tuple_0, str_0)
    var_0 = validation_0.to_try()
    var_1 = validation_0.to_either()
    var_2 = var_1.bind(tuple_0)
    var_0.to_either()


def test_case_4():
    set_0 = set()
    validation_0 = module_0.Validation(set_0, set_0)
    var_0 = validation_0.to_maybe()
    var_1 = var_0.to_box()
    var_2 = validation_0.is_fail()
    var_3 = validation_0.to_maybe()
    var_2.is_success()


def test_case_5():
    list_0 = []
    str_0 = "faslH%%*$T=ky"
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.to_maybe()
    var_1 = validation_0.to_try()
    list_0.is_success()


def test_case_6():
    list_0 = []
    list_0.is_success()


def test_case_7():
    bool_0 = True
    validation_0 = module_0.Validation(bool_0, bool_0)


def test_case_8():
    list_0 = []
    validation_0 = module_0.Validation(list_0, list_0)
    var_0 = validation_0.__str__()


def test_case_9():
    int_0 = 0
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    validation_0.map(int_0)


def test_case_10():
    bytes_0 = b"V\xfa\x12\xad\xb4F\xbc\xb7\x0f\xb3]\xf0\x17\x94^]\xf4"
    tuple_0 = (bytes_0,)
    set_0 = {tuple_0}
    bytes_1 = b"\xed\xe7\xbb"
    validation_0 = module_0.Validation(bytes_1, bytes_1)
    validation_0.bind(set_0)


def test_case_11():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    validation_0.ap(validation_0)


def test_case_12():
    bool_0 = True
    validation_0 = module_0.Validation(bool_0, bool_0)
    var_0 = validation_0.to_box()
    var_1 = bool_0.__str__()


def test_case_13():
    str_0 = ""
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    var_0 = validation_0.to_lazy()
    var_1 = var_0.to_either()
    var_2 = var_1.__eq__(str_0)
    var_3 = var_2.__str__()


def test_case_14():
    bool_0 = True
    validation_0 = module_0.Validation(bool_0, bool_0)
    validation_0.to_try()


def test_case_15():
    list_0 = []
    validation_0 = module_0.Validation(list_0, list_0)
    var_0 = validation_0.__eq__(validation_0)
    var_1 = validation_0.is_success()
    var_1.to_either()


def test_case_16():
    bytes_0 = b"xQ\xc1\xd0y\xe6l\xa7c\xe9"
    list_0 = []
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, bytes_0)
    var_0 = validation_0.to_try()
    validation_1 = module_0.Validation(var_0, list_0)
    var_1 = validation_1.__eq__(validation_0)
    var_2 = var_0.__eq__(list_0)
    bool_0 = False
    var_2.map(bool_0)
