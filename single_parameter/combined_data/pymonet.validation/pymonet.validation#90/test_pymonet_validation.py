# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.validation as module_0


def test_case_0():
    str_0 = "\n        Transform Validation to Try.\n\n        :returns: Lazy monad with function returning Validation value\n        :rtype: Lazy[Function() -> (A | None)]\n        "
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.__eq__(validation_0)
    var_1 = validation_0.__str__()
    bytes_0 = b"\x8d"
    validation_1 = module_0.Validation(var_1, bytes_0)
    validation_2 = module_0.Validation(var_1, var_1)
    var_2 = validation_2.is_fail()
    validation_2.bind(validation_2)


def test_case_1():
    bytes_0 = b"\xdd8_h\xe7jdLx"
    dict_0 = {bytes_0: bytes_0, bytes_0: bytes_0}
    validation_0 = module_0.Validation(bytes_0, bytes_0)
    var_0 = validation_0.to_try()
    var_1 = validation_0.__eq__(var_0)
    validation_1 = module_0.Validation(dict_0, bytes_0)
    var_2 = validation_1.to_try()
    var_3 = var_2.__str__()
    var_3.to_maybe()


def test_case_2():
    set_0 = set()
    validation_0 = module_0.Validation(set_0, set_0)
    var_0 = validation_0.to_either()
    var_1 = validation_0.to_lazy()
    var_2 = validation_0.to_box()
    var_2.is_fail()


def test_case_3():
    bool_0 = True
    validation_0 = module_0.Validation(bool_0, bool_0)


def test_case_4():
    list_0 = []
    validation_0 = module_0.Validation(list_0, list_0)
    var_0 = validation_0.to_lazy()
    var_1 = var_0.to_maybe()
    var_2 = validation_0.__str__()


def test_case_5():
    int_0 = 1149
    validation_0 = module_0.Validation(int_0, int_0)
    validation_0.is_fail()


def test_case_6():
    none_type_0 = None
    int_0 = -958
    validation_0 = module_0.Validation(int_0, int_0)
    validation_0.map(none_type_0)


def test_case_7():
    tuple_0 = ()
    validation_0 = module_0.Validation(tuple_0, tuple_0)
    float_0 = -523.4
    validation_0.bind(float_0)


def test_case_8():
    float_0 = -774.07
    bytes_0 = b"\xa2\xe7[\xfa\x8ad\x9d"
    validation_0 = module_0.Validation(bytes_0, bytes_0)
    validation_0.ap(float_0)


def test_case_9():
    list_0 = []
    validation_0 = module_0.Validation(list_0, list_0)
    var_0 = validation_0.to_box()
    var_1 = var_0.to_either()
    var_2 = var_1.to_maybe()
    var_3 = var_2.to_lazy()


def test_case_10():
    bool_0 = False
    tuple_0 = (bool_0,)
    str_0 = "6n"
    validation_0 = module_0.Validation(tuple_0, str_0)
    var_0 = validation_0.to_maybe()
    var_1 = var_0.to_try()


def test_case_11():
    str_0 = "\n        Transform Validation to Try.\n\n        :returns: Lazy monad with function returning Validation value\n        :rtype: Lazy[Function() -> (A | None)]\n        "
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.is_success()
    var_1 = validation_0.__eq__(validation_0)
    var_2 = validation_0.__str__()
    validation_1 = module_0.Validation(var_2, var_2)
    bool_0 = False
    var_3 = validation_1.__str__()
    tuple_0 = (bool_0,)
    validation_2 = module_0.Validation(var_2, tuple_0)
    var_4 = validation_2.is_fail()
    var_5 = validation_1.to_either()
    var_6 = validation_2.__eq__(var_1)
    var_7 = validation_1.to_maybe()
    var_1.ap(var_4)


def test_case_12():
    str_0 = "\n        Transform Validation to Try.\n\n        :returns: Lazy monad with function returning Validation value\n        :rtype: Lazy[Function() -> (A | None)]\nu       "
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.to_maybe()
    var_1 = validation_0.to_maybe()
    var_2 = validation_0.__eq__(validation_0)
    bytes_0 = b"\x8d"
    validation_1 = module_0.Validation(str_0, bytes_0)
    var_3 = validation_0.__eq__(validation_1)
    validation_1.map(str_0)


def test_case_13():
    tuple_0 = ()
    tuple_1 = (tuple_0,)
    float_0 = -1116.6846
    bytes_0 = b""
    dict_0 = {float_0: float_0, float_0: float_0, bytes_0: float_0}
    validation_0 = module_0.Validation(dict_0, bytes_0)
    var_0 = validation_0.to_maybe()
    var_1 = var_0.__eq__(tuple_1)
