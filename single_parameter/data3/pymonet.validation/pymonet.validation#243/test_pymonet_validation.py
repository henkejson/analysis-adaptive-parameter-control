# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.validation as module_0
import builtins as module_1


def test_case_0():
    tuple_0 = ()
    validation_0 = module_0.Validation(tuple_0, tuple_0)
    var_0 = validation_0.__eq__(validation_0)
    tuple_0.is_fail()


def test_case_1():
    bool_0 = True
    int_0 = 1
    validation_0 = module_0.Validation(int_0, int_0)
    validation_1 = module_0.Validation(validation_0, validation_0)
    var_0 = validation_1.__eq__(bool_0)
    var_0.is_success()


def test_case_2():
    bytes_0 = b"\x1c\x8a\xba\x0fW\t\x7f`\x15\x87\x81\xff\x02\xbc\r"
    validation_0 = module_0.Validation(bytes_0, bytes_0)
    var_0 = validation_0.__str__()
    object_0 = module_1.object()
    var_0.is_fail()


def test_case_3():
    int_0 = 1
    set_0 = {int_0, int_0}
    none_type_0 = None
    validation_0 = module_0.Validation(int_0, set_0)
    var_0 = validation_0.to_either()
    var_1 = var_0.bind(none_type_0)
    var_1.is_fail()


def test_case_4():
    tuple_0 = ()
    dict_0 = {}
    validation_0 = module_0.Validation(tuple_0, dict_0)
    validation_1 = module_0.Validation(validation_0, tuple_0)
    var_0 = validation_1.to_box()
    var_1 = validation_0.to_maybe()
    var_1.is_fail()


def test_case_5():
    none_type_0 = None
    str_0 = "cjW;GR_"
    validation_0 = module_0.Validation(none_type_0, str_0)
    var_0 = validation_0.to_maybe()
    var_1 = var_0.to_try()
    var_1.to_try()


def test_case_6():
    str_0 = "\n        Applies the function inside the Either[A] structure to another applicative type.\n\n        :param applicative: applicative contains function\n        :type applicative: Either[B]\n        :returns: new Either with result of contains function\n        :rtype: Either[A(B)]\n        "
    str_0.bind(str_0)


def test_case_7():
    object_0 = module_1.object()
    validation_0 = module_0.Validation(object_0, object_0)


def test_case_8():
    none_type_0 = None
    bool_0 = False
    validation_0 = module_0.Validation(none_type_0, bool_0)
    validation_0.is_success()


def test_case_9():
    bool_0 = False
    none_type_0 = None
    validation_0 = module_0.Validation(bool_0, none_type_0)
    validation_0.is_fail()


def test_case_10():
    none_type_0 = None
    bool_0 = False
    validation_0 = module_0.Validation(bool_0, bool_0)
    validation_0.map(none_type_0)


def test_case_11():
    complex_0 = -3202.645 - 135.0914j
    validation_0 = module_0.Validation(complex_0, complex_0)
    validation_0.bind(validation_0)


def test_case_12():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    bool_0 = False
    validation_1 = module_0.Validation(bool_0, bool_0)
    var_0 = validation_0.__eq__(validation_1)
    validation_0.ap(none_type_0)


def test_case_13():
    bytes_0 = b"a"
    validation_0 = module_0.Validation(bytes_0, bytes_0)
    var_0 = validation_0.to_either()
    var_1 = validation_0.to_box()
    var_2 = var_1.__eq__(bytes_0)
    var_3 = var_1.__eq__(bytes_0)
    var_4 = var_0.to_try()
    var_2.to_lazy()


def test_case_14():
    int_0 = 1062
    validation_0 = module_0.Validation(int_0, int_0)
    var_0 = validation_0.to_lazy()


def test_case_15():
    bool_0 = False
    set_0 = {bool_0, bool_0}
    bytes_0 = b"t\xccalN\xc0"
    list_0 = [set_0, bool_0, bytes_0, bool_0]
    bool_1 = False
    set_1 = {bool_1, bool_1}
    bytes_1 = b"p\xdb%\x90\xdb\x8cl\xebd\xd2"
    tuple_0 = (set_1, bool_1, bytes_1)
    validation_0 = module_0.Validation(tuple_0, bytes_1)
    var_0 = validation_0.to_lazy()
    var_1 = var_0.map(list_0)
    var_2 = var_1.to_try()
    var_2.to_either()


def test_case_16():
    bytes_0 = b"\xd4\x91\xf7\x99\xbd\xf3J\xa5\x0b\xf74C^}f\x9c"
    validation_0 = module_0.Validation(bytes_0, bytes_0)
    var_0 = validation_0.to_try()
    validation_1 = module_0.Validation(var_0, var_0)
    var_1 = validation_1.__eq__(validation_1)
    bytes_0.to_maybe()


def test_case_17():
    tuple_0 = ()
    validation_0 = module_0.Validation(tuple_0, tuple_0)
    var_0 = validation_0.__str__()
    var_1 = validation_0.to_try()
    var_2 = validation_0.is_success()
    var_0.to_box()


def test_case_18():
    int_0 = 1
    set_0 = set()
    none_type_0 = None
    validation_0 = module_0.Validation(int_0, set_0)
    var_0 = validation_0.to_either()
    var_0.bind(none_type_0)
