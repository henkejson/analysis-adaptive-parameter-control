# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.validation as module_0


def test_case_0():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    var_0 = validation_0.__eq__(validation_0)
    validation_0.is_success()


def test_case_1():
    none_type_0 = None
    bool_0 = False
    validation_0 = module_0.Validation(bool_0, none_type_0)
    var_0 = validation_0.__eq__(none_type_0)
    var_0.to_lazy()


def test_case_2():
    set_0 = set()
    validation_0 = module_0.Validation(set_0, set_0)
    var_0 = validation_0.to_maybe()
    var_1 = validation_0.__str__()
    none_type_0 = None
    set_0.ap(none_type_0)


def test_case_3():
    str_0 = "8/i=lh^,\n;7_"
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.__str__()
    bytes_0 = b"Nw \xb6\xfeKLxl\x9a\x9f\x9d:o\x11\xb6"
    tuple_0 = (bytes_0,)
    validation_1 = module_0.Validation(tuple_0, tuple_0)
    var_1 = validation_1.is_fail()
    var_1.is_success()


def test_case_4():
    list_0 = []
    validation_0 = module_0.Validation(list_0, list_0)
    var_0 = validation_0.to_maybe()
    var_1 = validation_0.__eq__(validation_0)
    var_1.to_lazy()


def test_case_5():
    list_0 = []
    validation_0 = module_0.Validation(list_0, list_0)


def test_case_6():
    bool_0 = True
    none_type_0 = None
    validation_0 = module_0.Validation(bool_0, none_type_0)
    validation_0.to_maybe()


def test_case_7():
    list_0 = []
    validation_0 = module_0.Validation(list_0, list_0)
    var_0 = validation_0.is_fail()
    var_0.to_lazy()


def test_case_8():
    bool_0 = True
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    validation_1 = module_0.Validation(validation_0, none_type_0)
    validation_0.map(bool_0)


def test_case_9():
    bool_0 = False
    int_0 = 1197
    none_type_0 = None
    validation_0 = module_0.Validation(int_0, none_type_0)
    validation_0.bind(bool_0)


def test_case_10():
    tuple_0 = ()
    validation_0 = module_0.Validation(tuple_0, tuple_0)
    validation_0.ap(validation_0)


def test_case_11():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    var_0 = validation_0.to_box()
    var_1 = var_0.to_maybe()
    var_1.to_maybe()


def test_case_12():
    str_0 = "\n    The Either type represents values with two possibilities: B value of type Either[A, B] is either Left[A or Right[B]\n    But not both in the same time.\n    "
    tuple_0 = (str_0,)
    int_0 = 1922
    validation_0 = module_0.Validation(tuple_0, int_0)
    var_0 = validation_0.to_lazy()
    var_1 = var_0.__str__()
    var_1.to_either()


def test_case_13():
    bool_0 = True
    dict_0 = {}
    validation_0 = module_0.Validation(dict_0, dict_0)
    var_0 = validation_0.to_lazy()
    var_1 = var_0.bind(bool_0)
    var_1.to_either()


def test_case_14():
    bool_0 = True
    validation_0 = module_0.Validation(bool_0, bool_0)
    validation_0.to_try()


def test_case_15():
    set_0 = set()
    validation_0 = module_0.Validation(set_0, set_0)
    var_0 = validation_0.to_maybe()
    var_1 = validation_0.__eq__(validation_0)
    var_2 = validation_0.to_box()
    var_3 = validation_0.to_either()
    var_4 = validation_0.to_lazy()
    var_5 = var_3.to_lazy()
    var_1.is_fail()


def test_case_16():
    none_type_0 = None
    str_0 = "_Pz)#4xp"
    set_0 = {str_0, none_type_0}
    validation_0 = module_0.Validation(set_0, set_0)
    var_0 = validation_0.to_either()
    var_0.is_success()


def test_case_17():
    str_0 = "y?6H\r-,7p7qv{8p"
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.to_maybe()
    var_1 = validation_0.__eq__(str_0)
    validation_0.bind(str_0)


def test_case_18():
    set_0 = set()
    validation_0 = module_0.Validation(set_0, set_0)
    var_0 = validation_0.to_maybe()
    var_1 = validation_0.to_maybe()
    var_2 = module_0.Validation(set_0, var_1)
    var_3 = var_2.__eq__(validation_0)
    var_4 = validation_0.to_box()
    var_5 = var_4.__str__()
    var_5.to_lazy()
