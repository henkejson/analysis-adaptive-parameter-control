# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.validation as module_0


def test_case_0():
    int_0 = 0
    validation_0 = module_0.Validation(int_0, int_0)
    var_0 = validation_0.__eq__(int_0)
    var_1 = var_0.__str__()
    var_1.to_maybe()


def test_case_1():
    int_0 = -1922
    str_0 = "g.&kD[~2Ob,'c4Q"
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.__str__()
    validation_0.ap(int_0)


def test_case_2():
    bytes_0 = b"[\xb8\xc9\x1fY"
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, bytes_0)
    var_0 = validation_0.to_either()
    var_1 = var_0.to_box()
    var_2 = var_1.to_maybe()
    var_3 = var_0.to_box()
    var_4 = validation_0.to_maybe()


def test_case_3():
    complex_0 = -1000 + 140.746286j
    validation_0 = module_0.Validation(complex_0, complex_0)


def test_case_4():
    bool_0 = True
    validation_0 = module_0.Validation(bool_0, bool_0)
    validation_0.__str__()


def test_case_5():
    bool_0 = False
    validation_0 = module_0.Validation(bool_0, bool_0)
    validation_0.is_fail()


def test_case_6():
    bool_0 = True
    validation_0 = module_0.Validation(bool_0, bool_0)
    validation_0.map(validation_0)


def test_case_7():
    list_0 = []
    validation_0 = module_0.Validation(list_0, list_0)
    float_0 = 2201.0
    validation_1 = module_0.Validation(list_0, float_0)
    validation_1.bind(validation_0)


def test_case_8():
    int_0 = -1922
    str_0 = "g.&kD[~2Ob,'c4Q"
    validation_0 = module_0.Validation(str_0, str_0)
    validation_0.ap(int_0)


def test_case_9():
    str_0 = "\n    Last is a Monoid that will always return the lastest, value when 2 Last instances are combined.\n    "
    validation_0 = module_0.Validation(str_0, str_0)
    none_type_0 = None
    int_0 = -736
    var_0 = validation_0.__eq__(none_type_0)
    validation_1 = module_0.Validation(int_0, int_0)
    var_1 = validation_0.__str__()
    var_2 = validation_0.__str__()
    var_3 = validation_0.to_box()


def test_case_10():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    var_0 = validation_0.to_lazy()
    var_1 = validation_0.__eq__(validation_0)
    validation_0.__str__()


def test_case_11():
    float_0 = 1068.00117
    bool_0 = False
    bool_1 = False
    validation_0 = module_0.Validation(bool_0, bool_1)
    var_0 = validation_0.to_lazy()
    var_1 = var_0.to_try()
    var_2 = var_1.__eq__(float_0)
    var_2.to_try()


def test_case_12():
    float_0 = -661.9707
    validation_0 = module_0.Validation(float_0, float_0)
    validation_0.to_try()


def test_case_13():
    float_0 = -661.9707
    validation_0 = module_0.Validation(float_0, float_0)
    var_0 = validation_0.__eq__(validation_0)
    validation_0.to_try()


def test_case_14():
    tuple_0 = ()
    validation_0 = module_0.Validation(tuple_0, tuple_0)
    validation_1 = module_0.Validation(tuple_0, tuple_0)
    var_0 = validation_1.to_box()
    var_1 = validation_1.to_try()
    var_2 = validation_0.to_box()
    var_3 = validation_1.__str__()
    var_4 = validation_1.is_success()
    var_5 = validation_1.to_lazy()
    var_4.is_fail()


def test_case_15():
    bool_0 = False
    set_0 = set()
    validation_0 = module_0.Validation(set_0, set_0)
    var_0 = validation_0.to_either()
    var_0.map(bool_0)


def test_case_16():
    bytes_0 = b"[\xb8\xc9\x1fY"
    dict_0 = {bytes_0: bytes_0, bytes_0: bytes_0, bytes_0: bytes_0, bytes_0: bytes_0}
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, dict_0)
    var_0 = validation_0.to_maybe()
    var_1 = var_0.bind(bytes_0)
    var_2 = var_1.to_try()
    validation_1 = module_0.Validation(none_type_0, bytes_0)
    var_3 = validation_1.__eq__(validation_0)
    validation_2 = module_0.Validation(dict_0, bytes_0)
    var_4 = validation_2.__eq__(var_0)
    var_5 = validation_1.to_maybe()
    bool_0 = False
    var_6 = validation_1.__eq__(bool_0)
    var_7 = validation_2.to_maybe()
    var_8 = var_7.__eq__(none_type_0)
    var_2.ap(var_8)


def test_case_17():
    bool_0 = True
    dict_0 = {}
    validation_0 = module_0.Validation(dict_0, dict_0)
    var_0 = validation_0.to_maybe()
    var_0.ap(bool_0)
