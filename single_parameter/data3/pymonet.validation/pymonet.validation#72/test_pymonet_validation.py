# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.validation as module_0


def test_case_0():
    float_0 = -1318.12636
    str_0 = "#&HXI"
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.__eq__(float_0)
    var_0.to_either()


def test_case_1():
    float_0 = -1318.12636
    str_0 = "#&HXI"
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.__eq__(float_0)
    var_1 = validation_0.__str__()
    var_1.to_either()


def test_case_2():
    none_type_0 = None
    set_0 = set()
    validation_0 = module_0.Validation(set_0, set_0)
    var_0 = validation_0.to_maybe()
    var_1 = var_0.__eq__(none_type_0)
    var_1.is_success()


def test_case_3():
    str_0 = "#&HXI"
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.__str__()
    var_1 = validation_0.to_maybe()
    var_2 = var_1.map(var_0)
    var_1.is_fail()


def test_case_4():
    bytes_0 = b"!\x80\x16O2\xabq\xe8\x9cAO\xae\xac\x01\x16\xab\x0b\xf4|-"
    bytes_0.bind(bytes_0)


def test_case_5():
    bytes_0 = b"!\x80\x16O2\xabq\xe8\x9cAO\xae\xac\x01\x16\xab\x0b\xf4|-"
    var_0 = module_0.Validation(bytes_0, bytes_0)


def test_case_6():
    str_0 = "#&HXI"
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.to_either()


def test_case_7():
    bool_0 = True
    validation_0 = module_0.Validation(bool_0, bool_0)
    validation_0.is_fail()


def test_case_8():
    none_type_0 = None
    int_0 = 1
    validation_0 = module_0.Validation(int_0, int_0)
    validation_0.map(none_type_0)


def test_case_9():
    str_0 = "(Z+5.TG}|ccFAf_"
    none_type_0 = None
    str_1 = "#&HXI"
    validation_0 = module_0.Validation(str_0, none_type_0)
    var_0 = validation_0.__eq__(validation_0)
    validation_0.bind(str_1)


def test_case_10():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    validation_0.ap(none_type_0)


def test_case_11():
    float_0 = -1318.12636
    str_0 = "#&HXI"
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.__eq__(float_0)
    var_1 = validation_0.to_box()
    var_2 = validation_0.__str__()
    validation_1 = module_0.Validation(var_0, var_1)
    var_3 = validation_0.__str__()
    var_3.to_either()


def test_case_12():
    str_0 = "Box[U]"
    none_type_0 = None
    validation_0 = module_0.Validation(str_0, none_type_0)
    var_0 = validation_0.to_lazy()
    float_0 = -1318.12636
    str_1 = "#&HXI"
    validation_1 = module_0.Validation(str_1, str_1)
    var_1 = validation_1.__eq__(float_0)
    var_2 = validation_1.__str__()
    var_2.to_either()


def test_case_13():
    bool_0 = True
    validation_0 = module_0.Validation(bool_0, bool_0)
    var_0 = validation_0.to_lazy()
    var_1 = var_0.map(validation_0)
    var_2 = var_0.to_box()
    var_0.to_lazy()


def test_case_14():
    bool_0 = True
    none_type_0 = None
    validation_0 = module_0.Validation(bool_0, none_type_0)
    validation_0.to_try()


def test_case_15():
    str_0 = "(Z+5.TG}|ccFAf_"
    none_type_0 = None
    validation_0 = module_0.Validation(str_0, none_type_0)
    var_0 = validation_0.__eq__(validation_0)
    var_0.to_either()


def test_case_16():
    list_0 = []
    validation_0 = module_0.Validation(list_0, list_0)
    var_0 = validation_0.__str__()
    var_0.ap(list_0)


def test_case_17():
    bytes_0 = b"!\x80\x16O2\xabq\xe8\x9cAO\xae\xac\x01\x16\xab\x0b\xf4|-"
    int_0 = -2937
    validation_0 = module_0.Validation(bytes_0, int_0)
    var_0 = validation_0.to_box()
    validation_1 = module_0.Validation(bytes_0, bytes_0)
    var_1 = validation_1.__eq__(bytes_0)
    var_2 = validation_1.__str__()
    var_3 = validation_1.__eq__(validation_0)
    validation_0.__str__()


def test_case_18():
    none_type_0 = None
    set_0 = set()
    validation_0 = module_0.Validation(set_0, set_0)
    var_0 = validation_0.to_maybe()
    var_1 = var_0.__eq__(none_type_0)
    var_2 = validation_0.to_either()
    var_2.to_either()
