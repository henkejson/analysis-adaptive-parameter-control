# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.validation as module_0
import builtins as module_1


def test_case_0():
    str_0 = "X50e%g!@[^@\x0c3Jr8"
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.__eq__(validation_0)
    var_0.to_lazy()


def test_case_1():
    str_0 = "X50e%g!@[^@\x0c3Jr8"
    validation_0 = module_0.Validation(str_0, str_0)
    none_type_0 = None
    var_0 = validation_0.__eq__(none_type_0)
    var_1 = validation_0.to_either()
    var_1.is_success()


def test_case_2():
    bool_0 = True
    dict_0 = {}
    validation_0 = module_0.Validation(dict_0, dict_0)
    var_0 = validation_0.__str__()
    var_1 = var_0.__eq__(bool_0)
    var_1.to_try()


def test_case_3():
    bool_0 = False
    bool_1 = False
    set_0 = {bool_1, bool_1, bool_1, bool_1}
    str_0 = "mrM"
    validation_0 = module_0.Validation(set_0, str_0)
    var_0 = validation_0.to_either()
    var_1 = var_0.bind(bool_0)
    var_2 = var_1.to_lazy()
    var_2.is_success()


def test_case_4():
    bool_0 = False
    bool_1 = False
    set_0 = {bool_1, bool_1, bool_1, bool_1}
    str_0 = "mrM"
    validation_0 = module_0.Validation(set_0, str_0)
    var_0 = validation_0.to_maybe()
    var_1 = var_0.map(validation_0)
    var_2 = validation_0.to_either()
    var_3 = var_2.bind(bool_0)
    var_4 = var_3.to_lazy()
    var_4.is_success()


def test_case_5():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)


def test_case_6():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    validation_0.to_try()


def test_case_7():
    bytes_0 = b"-\x19\xdd\xef\x14\x975\x7f\xcd\x87,\xb7\x12"
    validation_0 = module_0.Validation(bytes_0, bytes_0)
    validation_1 = module_0.Validation(bytes_0, validation_0)
    validation_1.is_fail()


def test_case_8():
    str_0 = "X50e%g!@[^@\x0c3Jr8"
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.is_success()
    none_type_0 = None
    var_1 = validation_0.__eq__(none_type_0)
    var_2 = validation_0.to_either()
    validation_0.map(var_0)


def test_case_9():
    int_0 = -4202
    bytes_0 = b"\x8c=\xbc\x85\xad@\xaf\xf9\xbe\xf8\x0b\xcd\xbdC\xa9Q\x99\xe0"
    validation_0 = module_0.Validation(bytes_0, bytes_0)
    validation_0.bind(int_0)


def test_case_10():
    bytes_0 = b"\xfaS"
    str_0 = "rm-HNE('l\rN1Y="
    validation_0 = module_0.Validation(str_0, str_0)
    validation_0.ap(bytes_0)


def test_case_11():
    bytes_0 = b"\xb1\xac#\x92G8O\xff\xf4F\x16Z\xc4.sE\xa6"
    tuple_0 = (bytes_0,)
    validation_0 = module_0.Validation(tuple_0, bytes_0)
    str_0 = "R9w6U/"
    validation_1 = module_0.Validation(str_0, str_0)
    var_0 = validation_1.to_box()
    var_0.ap(str_0)


def test_case_12():
    str_0 = "X50e%g!@[^@\x0c3Jr8"
    validation_0 = module_0.Validation(str_0, str_0)
    none_type_0 = None
    var_0 = validation_0.__eq__(none_type_0)
    var_1 = validation_0.to_lazy()
    var_1.is_success()


def test_case_13():
    bytes_0 = b"\xc9\xc7#\xd8"
    validation_0 = module_0.Validation(bytes_0, bytes_0)
    var_0 = validation_0.__eq__(validation_0)
    var_1 = validation_0.to_lazy()
    var_2 = var_1.to_either()
    var_3 = validation_0.is_success()
    var_0.is_fail()


def test_case_14():
    str_0 = "X50e%g!@[^@\x0c3Jr8"
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.__str__()
    var_1 = validation_0.is_success()
    none_type_0 = None
    var_2 = validation_0.__eq__(none_type_0)
    var_3 = validation_0.to_either()
    var_3.is_success()


def test_case_15():
    float_0 = -2186.5837
    bool_0 = False
    bytes_0 = b"\x0c\x064o>\xa7s\xa8\x9d\xcepp^\xbf\xde"
    bytes_1 = b""
    tuple_0 = (bool_0, bytes_0, bool_0, bytes_1)
    tuple_1 = (float_0, float_0, tuple_0, bytes_1)
    set_0 = {tuple_1, float_0, bytes_0}
    validation_0 = module_0.Validation(set_0, bytes_1)
    var_0 = validation_0.to_either()
    var_0.to_either()


def test_case_16():
    str_0 = "X50e%g!@[^@\x0c3Jr8"
    validation_0 = module_0.Validation(str_0, str_0)
    none_type_0 = None
    validation_1 = module_0.Validation(str_0, none_type_0)
    var_0 = validation_1.__eq__(validation_0)
    validation_1.to_try()


def test_case_17():
    float_0 = -2940.8
    list_0 = []
    object_0 = module_1.object(*list_0)
    validation_0 = module_0.Validation(object_0, list_0)
    var_0 = validation_0.to_maybe()
    var_1 = validation_0.__eq__(float_0)
    var_2 = validation_0.to_lazy()
    var_3 = var_2.to_try()
    var_3.to_maybe()
