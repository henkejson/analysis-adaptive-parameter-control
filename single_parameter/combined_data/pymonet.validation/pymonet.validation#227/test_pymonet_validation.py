# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.validation as module_0


def test_case_0():
    set_0 = set()
    validation_0 = module_0.Validation(set_0, set_0)
    var_0 = validation_0.__eq__(validation_0)
    var_1 = validation_0.to_box()
    var_2 = validation_0.to_box()


def test_case_1():
    str_0 = 'R\t#Z}3Yx#"r+QPFW1bO'
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.to_maybe()
    var_1 = validation_0.__eq__(str_0)
    var_1.to_box()


def test_case_2():
    tuple_0 = ()
    validation_0 = module_0.Validation(tuple_0, tuple_0)
    var_0 = validation_0.is_success()
    var_1 = validation_0.to_maybe()
    var_2 = validation_0.__str__()
    var_2.to_lazy()


def test_case_3():
    set_0 = set()
    validation_0 = module_0.Validation(set_0, set_0)
    var_0 = validation_0.to_either()
    var_1 = validation_0.to_box()
    var_0.is_success()


def test_case_4():
    str_0 = "b?\tWD.Ados#[fgW4/zv-"
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, str_0)
    var_0 = validation_0.to_either()
    var_1 = var_0.to_maybe()
    var_2 = var_1.ap(none_type_0)
    var_3 = var_2.to_try()
    validation_1 = module_0.Validation(str_0, none_type_0)


def test_case_5():
    set_0 = set()
    set_0.to_box()


def test_case_6():
    set_0 = set()
    var_0 = module_0.Validation(set_0, set_0)


def test_case_7():
    bool_0 = False
    validation_0 = module_0.Validation(bool_0, bool_0)
    validation_0.to_maybe()


def test_case_8():
    tuple_0 = ()
    validation_0 = module_0.Validation(tuple_0, tuple_0)
    validation_1 = module_0.Validation(tuple_0, validation_0)
    var_0 = validation_0.is_fail()
    var_0.to_box()


def test_case_9():
    float_0 = 1913.0
    int_0 = -1232
    list_0 = [int_0, int_0]
    none_type_0 = None
    validation_0 = module_0.Validation(list_0, none_type_0)
    validation_0.map(float_0)


def test_case_10():
    set_0 = set()
    validation_0 = module_0.Validation(set_0, set_0)
    var_0 = validation_0.__eq__(validation_0)
    validation_1 = module_0.Validation(set_0, validation_0)
    var_1 = validation_1.to_box()
    validation_1.bind(var_0)


def test_case_11():
    str_0 = "CE`5W:DDN"
    validation_0 = module_0.Validation(str_0, str_0)
    validation_0.ap(validation_0)


def test_case_12():
    set_0 = set()
    validation_0 = module_0.Validation(set_0, set_0)
    var_0 = validation_0.__eq__(validation_0)
    validation_1 = validation_0.to_lazy()
    var_1 = validation_1.to_box()
    var_2 = var_1.to_either()


def test_case_13():
    bool_0 = True
    validation_0 = module_0.Validation(bool_0, bool_0)
    validation_0.to_try()


def test_case_14():
    str_0 = 'R\t#Z}3Yx#"r+QPFW1bO'
    bool_0 = False
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.to_lazy()
    var_1 = bool_0.__eq__(validation_0)
    var_2 = validation_0.__str__()
    var_3 = validation_0.__eq__(var_1)
    var_0.is_success()


def test_case_15():
    set_0 = set()
    validation_0 = module_0.Validation(set_0, set_0)
    none_type_0 = None
    validation_1 = module_0.Validation(validation_0, none_type_0)
    var_0 = validation_1.__eq__(none_type_0)
    validation_2 = module_0.Validation(none_type_0, validation_1)
    var_1 = validation_2.__eq__(validation_0)
    var_0.to_box()
