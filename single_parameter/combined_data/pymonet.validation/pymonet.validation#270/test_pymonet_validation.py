# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.validation as module_0


def test_case_0():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    var_0 = validation_0.__eq__(none_type_0)
    var_0.ap(none_type_0)


def test_case_1():
    int_0 = -1726
    dict_0 = {int_0: int_0, int_0: int_0}
    float_0 = 179.0071
    dict_1 = {float_0: float_0, float_0: float_0, float_0: float_0, float_0: float_0}
    validation_0 = module_0.Validation(dict_1, dict_1)
    var_0 = validation_0.__str__()
    validation_0.map(dict_0)


def test_case_2():
    int_0 = -1726
    dict_0 = {int_0: int_0, int_0: int_0}
    dict_1 = {int_0: int_0, int_0: int_0, int_0: int_0, int_0: int_0}
    validation_0 = module_0.Validation(dict_1, dict_1)
    var_0 = validation_0.to_either()
    validation_0.map(dict_0)


def test_case_3():
    bool_0 = True
    validation_0 = module_0.Validation(bool_0, bool_0)
    bool_1 = True
    none_type_0 = None
    set_0 = {none_type_0, bool_1, bool_1}
    validation_1 = module_0.Validation(none_type_0, set_0)
    var_0 = validation_1.to_maybe()
    var_1 = var_0.map(bool_1)
    var_2 = var_1.bind(validation_0)
    var_3 = var_1.__str__()
    var_3.to_lazy()


def test_case_4():
    bytes_0 = b"H!\x15?gJH\xb9\xe7\x991\n\xb4\xb1\x99[\xb4\xd7"
    validation_0 = module_0.Validation(bytes_0, bytes_0)


def test_case_5():
    str_0 = "K_LXRj7V5>#n"
    set_0 = {str_0, str_0}
    float_0 = 1563.368
    tuple_0 = (set_0, float_0)
    str_1 = ""
    validation_0 = module_0.Validation(tuple_0, str_1)
    var_0 = validation_0.__str__()
    var_0.is_success()


def test_case_6():
    bool_0 = True
    validation_0 = module_0.Validation(bool_0, bool_0)
    validation_0.is_fail()


def test_case_7():
    bool_0 = False
    int_0 = 1
    validation_0 = module_0.Validation(int_0, int_0)
    validation_0.bind(bool_0)


def test_case_8():
    bool_0 = True
    bytes_0 = b"\xfa\xa3C\xcd\xb8\xb3\xef<i\x07\x834H\x83\xc4\x02>\xbe$\x1e"
    validation_0 = module_0.Validation(bytes_0, bytes_0)
    validation_0.ap(bool_0)


def test_case_9():
    bool_0 = True
    bool_1 = True
    validation_0 = module_0.Validation(bool_1, bool_1)
    var_0 = validation_0.to_box()
    var_0.ap(bool_0)


def test_case_10():
    str_0 = "n8AWvP.2x'n0\x0bY"
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    var_0 = validation_0.to_lazy()
    var_1 = var_0.ap(str_0)
    var_1.to_maybe()


def test_case_11():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    validation_1 = module_0.Validation(none_type_0, none_type_0)
    validation_1.to_try()


def test_case_12():
    none_type_0 = None
    set_0 = set()
    validation_0 = module_0.Validation(set_0, set_0)
    var_0 = validation_0.__eq__(validation_0)
    var_1 = validation_0.__str__()
    validation_1 = module_0.Validation(none_type_0, set_0)
    var_2 = validation_1.__eq__(validation_0)
    var_3 = validation_1.is_success()
    var_4 = validation_1.to_either()
    var_5 = validation_0.is_fail()
    var_6 = var_3.__str__()
    validation_2 = module_0.Validation(validation_0, set_0)
    var_7 = validation_2.is_fail()
    var_8 = var_3.__eq__(validation_1)
    var_9 = set_0.__str__()
    var_6.to_maybe()


def test_case_13():
    none_type_0 = None
    set_0 = set()
    validation_0 = module_0.Validation(set_0, set_0)
    var_0 = validation_0.to_lazy()
    validation_1 = module_0.Validation(none_type_0, set_0)
    var_1 = validation_1.__eq__(validation_0)
    var_2 = validation_1.is_success()
    var_3 = var_0.to_either()
    var_4 = validation_0.is_fail()
    var_5 = var_2.__str__()
    validation_2 = module_0.Validation(validation_0, set_0)
    var_6 = validation_0.to_maybe()
    str_0 = "Ru#Nd|;mb,^TS~R"
    var_5.map(str_0)


def test_case_14():
    none_type_0 = None
    set_0 = set()
    validation_0 = module_0.Validation(none_type_0, set_0)
    var_0 = validation_0.__eq__(validation_0)
    var_1 = validation_0.is_success()
    var_1.is_fail()


def test_case_15():
    int_0 = 560
    var_0 = module_0.Validation(int_0, int_0)
    var_1 = var_0.to_lazy()
    var_2 = var_1.to_maybe()
    none_type_0 = None
    none_type_1 = None
    validation_0 = module_0.Validation(int_0, var_2)
    validation_1 = module_0.Validation(none_type_1, none_type_1)
    validation_2 = module_0.Validation(none_type_0, none_type_1)
    var_3 = validation_1.to_box()
    var_4 = var_3.to_maybe()
    tuple_0 = (var_3, var_3)
    validation_3 = module_0.Validation(none_type_1, tuple_0)
    var_2.to_maybe()


def test_case_16():
    set_0 = set()
    validation_0 = module_0.Validation(set_0, set_0)
    var_0 = validation_0.to_lazy()
    int_0 = 1194
    validation_1 = module_0.Validation(validation_0, int_0)
    var_1 = validation_1.to_box()
    var_2 = validation_1.__eq__(validation_0)
    var_0.is_success()
