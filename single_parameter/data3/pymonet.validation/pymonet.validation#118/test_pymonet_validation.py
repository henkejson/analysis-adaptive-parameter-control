# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.validation as module_0
import builtins as module_1


def test_case_0():
    bool_0 = False
    validation_0 = module_0.Validation(bool_0, bool_0)
    var_0 = validation_0.__eq__(bool_0)


def test_case_1():
    bytes_0 = b"w\x8fyT\xec\xb7\x08\xaa\x8f_\t"
    dict_0 = {bytes_0: bytes_0, bytes_0: bytes_0, bytes_0: bytes_0, bytes_0: bytes_0}
    list_0 = [dict_0, dict_0, dict_0]
    list_1 = [list_0]
    validation_0 = module_0.Validation(list_1, list_0)
    var_0 = validation_0.to_either()
    var_1 = var_0.__eq__(list_0)
    var_2 = var_0.__str__()
    validation_1 = module_0.Validation(var_1, var_1)
    var_3 = validation_0.__eq__(var_0)
    var_4 = validation_0.__str__()
    var_5 = validation_0.__eq__(bytes_0)
    var_5.is_fail()


def test_case_2():
    str_0 = 'v6\t"&tOj%?7\r'
    set_0 = {str_0}
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, set_0)
    var_0 = validation_0.to_maybe()
    var_1 = var_0.map(str_0)
    var_2 = var_1.to_lazy()
    var_3 = var_2.map(str_0)
    var_4 = var_3.bind(str_0)
    var_5 = var_4.to_try()
    validation_1 = module_0.Validation(set_0, str_0)
    var_6 = validation_1.is_success()


def test_case_3():
    bytes_0 = b"M\x8fyT\xec\xb7\x08@\x8f_\t"
    validation_0 = module_0.Validation(bytes_0, bytes_0)


def test_case_4():
    int_0 = 621
    validation_0 = module_0.Validation(int_0, int_0)
    validation_0.to_maybe()


def test_case_5():
    bool_0 = True
    validation_0 = module_0.Validation(bool_0, bool_0)
    validation_0.is_fail()


def test_case_6():
    none_type_0 = None
    bytes_0 = b"\xe8.#\xb0["
    validation_0 = module_0.Validation(none_type_0, bytes_0)
    var_0 = validation_0.to_maybe()
    var_1 = var_0.to_box()
    validation_0.map(none_type_0)


def test_case_7():
    int_0 = 1
    validation_0 = module_0.Validation(int_0, int_0)
    validation_0.bind(validation_0)


def test_case_8():
    bytes_0 = b"5\xe7\xe5\xc7\xea\xe9*\xe1\x9c\xd1\xed\xf8u"
    int_0 = -2456
    dict_0 = {int_0: int_0}
    none_type_0 = None
    validation_0 = module_0.Validation(dict_0, none_type_0)
    validation_0.ap(bytes_0)


def test_case_9():
    bool_0 = False
    validation_0 = module_0.Validation(bool_0, bool_0)
    var_0 = validation_0.to_box()
    var_0.to_box()


def test_case_10():
    bytes_0 = b"w\x8fyT\xec\xb7\x08\xaa\x8f_\t"
    dict_0 = {bytes_0: bytes_0, bytes_0: bytes_0, bytes_0: bytes_0, bytes_0: bytes_0}
    list_0 = [dict_0, dict_0, dict_0]
    list_1 = [list_0]
    validation_0 = module_0.Validation(list_1, list_0)
    var_0 = validation_0.to_either()
    var_1 = var_0.__str__()
    bool_0 = False
    var_2 = validation_0.to_lazy()
    validation_1 = module_0.Validation(bool_0, bool_0)
    var_3 = var_0.to_maybe()
    var_0.to_either()


def test_case_11():
    bool_0 = False
    validation_0 = module_0.Validation(bool_0, bool_0)
    var_0 = validation_0.to_lazy()
    var_1 = var_0.to_maybe()
    var_1.to_maybe()


def test_case_12():
    object_0 = module_1.object()
    validation_0 = module_0.Validation(object_0, object_0)
    var_0 = validation_0.__eq__(validation_0)
    var_1 = var_0.__eq__(validation_0)
    validation_0.to_try()


def test_case_13():
    int_0 = 588
    validation_0 = module_0.Validation(int_0, int_0)
    var_0 = validation_0.__eq__(validation_0)
    var_0.to_maybe()


def test_case_14():
    bytes_0 = b"\xcb6\xc9H\xf9"
    validation_0 = module_0.Validation(bytes_0, bytes_0)
    var_0 = validation_0.to_either()
    var_1 = var_0.__eq__(bytes_0)
    var_1.to_maybe()


def test_case_15():
    object_0 = module_1.object()
    dict_0 = {}
    validation_0 = module_0.Validation(dict_0, dict_0)
    var_0 = validation_0.__str__()
    var_0.is_success()


def test_case_16():
    set_0 = set()
    validation_0 = module_0.Validation(set_0, set_0)
    var_0 = validation_0.to_maybe()
    var_1 = var_0.__str__()
    str_0 = "2"
    str_1 = "#\rz)DRAJu"
    var_2 = validation_0.__eq__(str_1)
    validation_1 = module_0.Validation(str_0, str_0)
    var_3 = validation_1.to_maybe()
    var_4 = validation_1.to_maybe()
    validation_1.map(var_4)


def test_case_17():
    bytes_0 = b"w\x8fyT\xec\xb7\x08\xaa\x8f_\t"
    dict_0 = {bytes_0: bytes_0, bytes_0: bytes_0, bytes_0: bytes_0, bytes_0: bytes_0}
    list_0 = [dict_0, dict_0, dict_0]
    list_1 = []
    validation_0 = module_0.Validation(list_1, list_0)
    var_0 = validation_0.to_either()
    var_1 = var_0.__str__()
    bool_0 = False
    validation_1 = module_0.Validation(bool_0, bool_0)
    var_2 = validation_1.__eq__(validation_0)
    bool_1 = False
    var_2.ap(bool_1)


def test_case_18():
    object_0 = module_1.object()
    dict_0 = {}
    validation_0 = module_0.Validation(dict_0, dict_0)
    var_0 = validation_0.__str__()
    var_1 = validation_0.to_either()
    var_0.is_success()
