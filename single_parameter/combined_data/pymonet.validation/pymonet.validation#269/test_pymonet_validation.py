# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.validation as module_0


def test_case_0():
    bytes_0 = b"U\xa0:\t\xd7\xd8\xbd\xba\x1e\x8el\xde"
    validation_0 = module_0.Validation(bytes_0, bytes_0)
    var_0 = validation_0.__str__()
    bytes_1 = b"\x15"
    var_1 = validation_0.__eq__(bytes_1)
    validation_1 = module_0.Validation(bytes_0, var_1)
    var_2 = var_0.__eq__(validation_0)
    var_3 = validation_0.to_lazy()
    var_0.bind(var_3)


def test_case_1():
    str_0 = "B7w1*y)'_whks/t5J"
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.to_either()
    bytes_0 = b"j\x8f\xa0\xe7EL\xdd\xde\xd1\x9f\x82\xadZvh\x1b"
    list_0 = [bytes_0, bytes_0]
    validation_1 = module_0.Validation(list_0, bytes_0)
    bool_0 = False
    var_1 = var_0.__eq__(var_0)
    validation_1.map(bool_0)


def test_case_2():
    dict_0 = {}
    validation_0 = module_0.Validation(dict_0, dict_0)
    var_0 = validation_0.to_maybe()
    bool_0 = True
    none_type_0 = None
    validation_1 = module_0.Validation(none_type_0, bool_0)
    validation_1.__str__()


def test_case_3():
    none_type_0 = None
    none_type_0.is_fail()


def test_case_4():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)


def test_case_5():
    int_0 = 4
    validation_0 = module_0.Validation(int_0, int_0)
    validation_0.to_try()


def test_case_6():
    dict_0 = {}
    validation_0 = module_0.Validation(dict_0, dict_0)
    var_0 = validation_0.to_lazy()
    float_0 = -1342.7816
    var_1 = validation_0.is_fail()
    validation_1 = module_0.Validation(float_0, float_0)


def test_case_7():
    none_type_0 = None
    bool_0 = True
    validation_0 = module_0.Validation(bool_0, bool_0)
    validation_0.map(none_type_0)


def test_case_8():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    validation_0.bind(validation_0)


def test_case_9():
    set_0 = set()
    validation_0 = module_0.Validation(set_0, set_0)
    var_0 = validation_0.__str__()
    str_0 = "5$r?x^VF"
    validation_1 = module_0.Validation(str_0, str_0)
    validation_1.ap(set_0)


def test_case_10():
    int_0 = 3296
    int_1 = 676
    str_0 = "pkpQbMEl=S"
    tuple_0 = (int_0, int_1, str_0, int_0)
    validation_0 = module_0.Validation(tuple_0, int_1)
    var_0 = validation_0.to_box()
    var_0.to_box()


def test_case_11():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    var_0 = validation_0.to_lazy()


def test_case_12():
    complex_0 = 3501.462628 + 1367.035j
    bool_0 = True
    validation_0 = module_0.Validation(bool_0, bool_0)
    var_0 = validation_0.to_lazy()
    var_1 = var_0.to_try()
    var_1.map(complex_0)


def test_case_13():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    var_0 = validation_0.__eq__(validation_0)
    var_1 = validation_0.to_lazy()


def test_case_14():
    tuple_0 = ()
    str_0 = "\n        Take mapper function and return new instance of Left with the same value.\n\n        :returns: Copy of self\n        :rtype: Left[A]\n        "
    validation_0 = module_0.Validation(tuple_0, tuple_0)
    var_0 = validation_0.to_either()
    var_1 = var_0.to_lazy()
    var_2 = var_1.to_try()
    var_3 = var_2.__eq__(str_0)
    var_3.to_either()


def test_case_15():
    bytes_0 = b'\x11:\x15(sM\x02g\x9fb\x1c\x1a\xd3\xc9\xb3\xbd7"\x19\x11'
    validation_0 = module_0.Validation(bytes_0, bytes_0)
    var_0 = validation_0.to_maybe()
    validation_0.map(var_0)


def test_case_16():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    validation_1 = module_0.Validation(none_type_0, validation_0)
    var_0 = validation_0.__eq__(validation_1)
    validation_2 = module_0.Validation(none_type_0, var_0)
    validation_3 = module_0.Validation(validation_0, none_type_0)
    validation_2.to_try()
