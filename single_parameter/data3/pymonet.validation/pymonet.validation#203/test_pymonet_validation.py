# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.validation as module_0


def test_case_0():
    int_0 = 196
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    var_0 = validation_0.__eq__(validation_0)
    var_1 = int_0.__str__()
    validation_0.to_maybe()


def test_case_1():
    str_0 = "3F#y"
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.to_box()
    var_1 = validation_0.__eq__(str_0)
    var_1.ap(str_0)


def test_case_2():
    tuple_0 = ()
    bytes_0 = b"k\x8f1\xba\x7f^*\x19\xa5\x8d\xb6"
    dict_0 = {tuple_0: tuple_0, tuple_0: tuple_0, tuple_0: bytes_0}
    validation_0 = module_0.Validation(dict_0, tuple_0)
    var_0 = validation_0.__str__()
    var_0.to_box()


def test_case_3():
    bytes_0 = b":A\xf4@"
    validation_0 = module_0.Validation(bytes_0, bytes_0)
    var_0 = validation_0.to_try()
    var_1 = validation_0.to_either()
    var_2 = validation_0.__str__()
    var_1.to_either()


def test_case_4():
    dict_0 = {}
    validation_0 = module_0.Validation(dict_0, dict_0)
    var_0 = validation_0.to_maybe()
    var_0.to_maybe()


def test_case_5():
    bytes_0 = b"\x89\xeee\x81M\x1a\xfboL\x16PW\x1a\xd6Y"
    validation_0 = module_0.Validation(bytes_0, bytes_0)
    var_0 = validation_0.to_maybe()


def test_case_6():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)


def test_case_7():
    bool_0 = False
    validation_0 = module_0.Validation(bool_0, bool_0)
    validation_0.is_success()


def test_case_8():
    bytes_0 = b'\xf5\xacK\xa5\xde\xf5\x94\xf6y\\X8\xbb|\x1c\xbb"'
    dict_0 = {bytes_0: bytes_0, bytes_0: bytes_0, bytes_0: bytes_0, bytes_0: bytes_0}
    validation_0 = module_0.Validation(dict_0, dict_0)
    var_0 = validation_0.to_either()
    var_1 = validation_0.is_fail()
    var_2 = var_0.to_lazy()
    var_3 = var_2.__str__()
    var_3.is_fail()


def test_case_9():
    tuple_0 = ()
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    validation_0.map(tuple_0)


def test_case_10():
    bytes_0 = b"\x7fvy\xb0]`\xc1\x9e\xeb\t"
    float_0 = -7.4065
    float_1 = 949.825686
    validation_0 = module_0.Validation(float_0, float_1)
    validation_0.bind(bytes_0)


def test_case_11():
    str_0 = ""
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0, str_0: str_0}
    validation_0 = module_0.Validation(str_0, dict_0)
    var_0 = validation_0.to_maybe()
    validation_0.ap(var_0)


def test_case_12():
    bool_0 = True
    validation_0 = module_0.Validation(bool_0, bool_0)
    var_0 = validation_0.to_box()


def test_case_13():
    str_0 = "qq@>Gtr4J!2B\towfo"
    none_type_0 = None
    validation_0 = module_0.Validation(str_0, none_type_0)
    var_0 = validation_0.to_lazy()


def test_case_14():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    var_0 = validation_0.to_lazy()
    var_1 = var_0.to_try()
    var_1.map(none_type_0)


def test_case_15():
    bytes_0 = b":A\xf4@"
    validation_0 = module_0.Validation(bytes_0, bytes_0)
    var_0 = validation_0.to_either()
    var_1 = module_0.Validation(validation_0, validation_0)
    var_2 = validation_0.__eq__(var_1)
    var_3 = var_1.__eq__(var_0)
    var_2.to_either()


def test_case_16():
    set_0 = set()
    validation_0 = module_0.Validation(set_0, set_0)
    var_0 = validation_0.to_either()
    var_1 = var_0.__eq__(set_0)
    var_1.bind(set_0)
