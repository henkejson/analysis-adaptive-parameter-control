# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.validation as module_0


def test_case_0():
    bool_0 = False
    int_0 = 1626
    validation_0 = module_0.Validation(int_0, int_0)
    var_0 = validation_0.__eq__(bool_0)
    var_0.to_either()


def test_case_1():
    str_0 = "EPMZ"
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0}
    validation_0 = module_0.Validation(str_0, dict_0)
    var_0 = validation_0.to_either()


def test_case_2():
    none_type_0 = None
    str_0 = ""
    validation_0 = module_0.Validation(none_type_0, str_0)
    var_0 = validation_0.__str__()
    var_1 = validation_0.to_maybe()
    dict_0 = {}
    validation_1 = module_0.Validation(dict_0, dict_0)
    var_2 = validation_1.to_lazy()
    dict_1 = {}
    validation_1.bind(dict_1)


def test_case_3():
    none_type_0 = None
    str_0 = "LgMx!D'&6%"
    set_0 = {str_0, str_0}
    validation_0 = module_0.Validation(set_0, set_0)
    var_0 = validation_0.to_maybe()
    validation_0.bind(none_type_0)


def test_case_4():
    int_0 = 1626
    validation_0 = module_0.Validation(int_0, int_0)


def test_case_5():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    validation_0.to_either()


def test_case_6():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    validation_0.is_fail()


def test_case_7():
    float_0 = 3431.38
    validation_0 = module_0.Validation(float_0, float_0)
    int_0 = 0
    validation_1 = module_0.Validation(int_0, int_0)
    var_0 = validation_1.to_lazy()
    var_1 = var_0.to_either()
    validation_1.map(validation_1)


def test_case_8():
    set_0 = set()
    str_0 = "1VJ<+"
    none_type_0 = None
    validation_0 = module_0.Validation(str_0, none_type_0)
    validation_0.bind(set_0)


def test_case_9():
    none_type_0 = None
    int_0 = 163
    validation_0 = module_0.Validation(none_type_0, int_0)
    var_0 = validation_0.__eq__(validation_0)
    validation_0.ap(validation_0)


def test_case_10():
    bool_0 = True
    validation_0 = module_0.Validation(bool_0, bool_0)
    var_0 = validation_0.to_box()
    var_1 = var_0.to_try()
    var_1.to_try()


def test_case_11():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    validation_0.to_try()


def test_case_12():
    none_type_0 = None
    int_0 = 163
    validation_0 = module_0.Validation(int_0, none_type_0)
    var_0 = validation_0.__eq__(validation_0)
    validation_0.is_fail()


def test_case_13():
    none_type_0 = None
    int_0 = 163
    validation_0 = module_0.Validation(none_type_0, int_0)
    validation_1 = module_0.Validation(validation_0, validation_0)
    var_0 = validation_0.__eq__(validation_1)
    validation_1.to_either()


def test_case_14():
    str_0 = "[Ic`@}"
    dict_0 = {}
    validation_0 = module_0.Validation(dict_0, str_0)
    var_0 = validation_0.to_maybe()
    validation_1 = module_0.Validation(str_0, str_0)
    var_1 = validation_1.to_lazy()
    var_2 = validation_1.__str__()
    var_3 = validation_1.is_fail()
    var_4 = validation_1.to_box()
    var_2.to_either()


def test_case_15():
    str_0 = ""
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.to_maybe()
    var_1 = var_0.to_either()
    var_2 = validation_0.to_lazy()
    var_3 = validation_0.to_maybe()
    validation_1 = module_0.Validation(var_1, var_1)
    var_4 = validation_0.to_either()
    var_5 = var_4.to_box()
    var_6 = var_2.to_box()
    var_7 = var_6.ap(var_2)
    var_8 = var_2.map(var_6)
    int_0 = 0
    none_type_0 = None
    var_9 = var_2.bind(none_type_0)
    validation_2 = module_0.Validation(int_0, var_8)
    var_10 = validation_2.__eq__(var_2)
    var_6.is_fail()
