# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import builtins as module_0
import pymonet.validation as module_1


def test_case_0():
    object_0 = module_0.object()
    tuple_0 = ()
    validation_0 = module_1.Validation(tuple_0, tuple_0)
    var_0 = validation_0.to_either()
    int_0 = 1
    validation_1 = module_1.Validation(var_0, int_0)
    var_1 = validation_1.__eq__(object_0)


def test_case_1():
    bytes_0 = b";A\xea5?\xcb\xb7OggFF\xf833x\x9d"
    validation_0 = module_1.Validation(bytes_0, bytes_0)
    var_0 = validation_0.__str__()


def test_case_2():
    bool_0 = False
    list_0 = [bool_0, bool_0]
    set_0 = set()
    bytes_0 = b"VUsEl\xa1"
    dict_0 = {bytes_0: bytes_0, bytes_0: bytes_0, bytes_0: bytes_0}
    list_1 = [dict_0]
    validation_0 = module_1.Validation(list_1, bytes_0)
    var_0 = validation_0.is_fail()
    var_1 = validation_0.to_either()
    bool_1 = False
    var_2 = var_1.bind(bool_1)
    validation_1 = module_1.Validation(list_0, list_1)
    var_3 = var_0.__eq__(set_0)
    var_4 = var_3.__eq__(list_0)
    var_5 = var_1.ap(var_1)
    set_0.is_success()


def test_case_3():
    str_0 = "Box[value={}]"
    validation_0 = module_1.Validation(str_0, str_0)
    var_0 = validation_0.to_maybe()
    var_1 = var_0.__eq__(str_0)
    var_2 = var_0.to_lazy()
    var_3 = var_2.to_try()
    var_4 = var_0.to_lazy()
    var_5 = validation_0.is_fail()
    var_6 = var_0.to_either()


def test_case_4():
    bytes_0 = b";A\xea5?\xcb\xb7OggFF\xf833x\x9d"
    validation_0 = module_1.Validation(bytes_0, bytes_0)


def test_case_5():
    str_0 = "/_.8Oi|"
    validation_0 = module_1.Validation(str_0, str_0)
    var_0 = validation_0.is_fail()
    var_0.to_maybe()


def test_case_6():
    str_0 = "gjN3\ruT|bf=QRtGu!F"
    validation_0 = module_1.Validation(str_0, str_0)
    validation_0.map(validation_0)


def test_case_7():
    bytes_0 = b";A\xea5?\xcb\xb7OggFFI\xc8\xf833x\x9d"
    validation_0 = module_1.Validation(bytes_0, bytes_0)
    validation_0.bind(validation_0)


def test_case_8():
    bool_0 = False
    validation_0 = module_1.Validation(bool_0, bool_0)
    validation_0.ap(validation_0)


def test_case_9():
    object_0 = module_0.object()
    tuple_0 = ()
    validation_0 = module_1.Validation(tuple_0, tuple_0)
    validation_1 = module_1.Validation(object_0, validation_0)
    var_0 = validation_1.to_box()
    var_1 = validation_0.to_lazy()
    var_0.to_box()


def test_case_10():
    none_type_0 = None
    validation_0 = module_1.Validation(none_type_0, none_type_0)
    validation_0.to_try()


def test_case_11():
    bytes_0 = b";A\xea5?\xcb\xb7OggFF\xf833x\x9d"
    float_0 = 1796.1642017948113
    none_type_0 = None
    bool_0 = False
    validation_0 = module_1.Validation(bytes_0, none_type_0)
    var_0 = validation_0.__eq__(validation_0)
    validation_1 = module_1.Validation(float_0, float_0)
    validation_2 = module_1.Validation(bool_0, validation_0)
    var_0.bind(none_type_0)


def test_case_12():
    bytes_0 = b";A\xea5?\xcb\xb7OggFF\xf833x\x9d"
    float_0 = 1814.99593
    none_type_0 = None
    bool_0 = False
    validation_0 = module_1.Validation(none_type_0, bytes_0)
    var_0 = validation_0.to_maybe()
    var_1 = var_0.map(none_type_0)
    validation_1 = module_1.Validation(bytes_0, bytes_0)
    var_2 = validation_1.__eq__(none_type_0)
    validation_2 = module_1.Validation(validation_1, float_0)
    validation_3 = validation_1.__eq__(validation_2)
    validation_1.ap(bool_0)


def test_case_13():
    bytes_0 = b"E"
    bool_0 = False
    validation_0 = module_1.Validation(bool_0, bool_0)
    var_0 = validation_0.to_lazy()
    var_1 = var_0.to_either()
    var_2 = var_0.bind(bytes_0)
    var_2.to_box()


def test_case_14():
    bytes_0 = b""
    none_type_0 = None
    validation_0 = module_1.Validation(bytes_0, bytes_0)
    var_0 = validation_0.to_maybe()
    var_1 = validation_0.__eq__(none_type_0)
    var_2 = var_1.__str__()
    var_3 = validation_0.is_fail()
    validation_1 = module_1.Validation(bytes_0, none_type_0)
    var_0.is_fail()


def test_case_15():
    object_0 = module_0.object()
    tuple_0 = ()
    validation_0 = module_1.Validation(tuple_0, tuple_0)
    var_0 = validation_0.__str__()
    int_0 = 1
    validation_1 = module_1.Validation(var_0, int_0)
    var_1 = validation_1.__eq__(object_0)
