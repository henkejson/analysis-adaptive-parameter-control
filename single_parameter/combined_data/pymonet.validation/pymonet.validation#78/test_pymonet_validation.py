# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.validation as module_0


def test_case_0():
    str_0 = "Rsrd\t{8*)(~ta.5"
    float_0 = 4202.78716
    validation_0 = module_0.Validation(float_0, str_0)
    var_0 = validation_0.__eq__(float_0)
    var_0.bind(float_0)


def test_case_1():
    bool_0 = True
    str_0 = "AM#!-"
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.__str__()
    var_1 = validation_0.is_fail()
    var_2 = validation_0.to_either()
    var_3 = validation_0.__eq__(str_0)
    validation_1 = module_0.Validation(var_2, var_2)
    var_4 = validation_1.__eq__(validation_1)
    var_5 = var_2.map(bool_0)


def test_case_2():
    list_0 = []
    validation_0 = module_0.Validation(list_0, list_0)
    var_0 = validation_0.to_maybe()
    var_1 = validation_0.to_either()
    var_1.ap(var_1)


def test_case_3():
    str_0 = "Rsrd\t{8*)(~ta.5"
    float_0 = 4202.78716
    validation_0 = module_0.Validation(float_0, str_0)
    var_0 = validation_0.__eq__(float_0)
    var_1 = validation_0.to_either()
    validation_0.ap(var_1)


def test_case_4():
    bytes_0 = b"\xe8\x13=M{g\xf4\xcf\xda\x8d\x94Zf\xf3\xe7e\xcbJk"
    validation_0 = module_0.Validation(bytes_0, bytes_0)


def test_case_5():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    validation_0.to_either()


def test_case_6():
    int_0 = -781
    validation_0 = module_0.Validation(int_0, int_0)
    validation_0.is_fail()


def test_case_7():
    bool_0 = True
    bool_1 = False
    str_0 = "AM#!-"
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.__str__()
    var_1 = validation_0.is_fail()
    var_2 = validation_0.to_either()
    var_3 = validation_0.__eq__(str_0)
    validation_1 = module_0.Validation(var_1, var_2)
    var_4 = validation_1.__eq__(validation_1)
    tuple_0 = (var_4, bool_1, var_2, bool_0)
    validation_0.map(tuple_0)


def test_case_8():
    float_0 = 825.827
    validation_0 = module_0.Validation(float_0, float_0)
    validation_0.bind(float_0)


def test_case_9():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    var_0 = validation_0.to_box()
    var_0.is_fail()


def test_case_10():
    int_0 = 1695
    list_0 = [int_0, int_0, int_0]
    bytes_0 = b""
    none_type_0 = None
    validation_0 = module_0.Validation(bytes_0, none_type_0)
    var_0 = validation_0.to_lazy()
    var_1 = var_0.to_box()
    var_2 = var_1.to_maybe()
    var_3 = var_2.to_lazy()
    var_4 = var_3.map(list_0)
    var_4.to_lazy()


def test_case_11():
    str_0 = "49T\x0bYHGA:"
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.to_try()
    var_1 = validation_0.__eq__(validation_0)
    var_2 = var_0.__str__()
    var_2.to_lazy()


def test_case_12():
    bool_0 = True
    bool_1 = False
    str_0 = "AM#!-"
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.is_fail()
    var_1 = validation_0.to_either()
    var_2 = validation_0.__eq__(str_0)
    var_3 = var_1.map(bool_1)
    var_4 = bool_0.__eq__(var_3)
    var_5 = validation_0.to_maybe()
    var_6 = var_5.__eq__(validation_0)
    var_7 = var_5.to_either()
    var_6.to_either()


def test_case_13():
    tuple_0 = ()
    validation_0 = module_0.Validation(tuple_0, tuple_0)
    var_0 = validation_0.__str__()
    var_1 = var_0.__str__()
    var_2 = validation_0.__str__()
    var_3 = validation_0.to_box()
    var_4 = validation_0.to_either()
    var_2.is_success()


def test_case_14():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    var_0 = validation_0.__eq__(validation_0)
    var_0.is_success()


def test_case_15():
    str_0 = "A?#M-"
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.to_try()
    var_1 = validation_0.to_either()
    var_2 = validation_0.to_lazy()
    var_3 = validation_0.__eq__(str_0)
    validation_1 = module_0.Validation(var_0, var_1)
    var_4 = validation_1.__eq__(validation_1)
    var_5 = var_2.to_maybe()
    var_6 = validation_1.__eq__(validation_0)
    var_7 = validation_0.to_either()
    var_6.to_either()
