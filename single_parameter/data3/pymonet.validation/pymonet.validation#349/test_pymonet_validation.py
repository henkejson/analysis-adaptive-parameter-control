# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.validation as module_0
import builtins as module_1


def test_case_0():
    bool_0 = False
    validation_0 = module_0.Validation(bool_0, bool_0)
    bytes_0 = b"\xd4%O\x9d\xa4\xa8\xeeJ\xc3o\x0e"
    validation_1 = module_0.Validation(bytes_0, bytes_0)
    var_0 = validation_1.to_maybe()
    validation_2 = module_0.Validation(bytes_0, bytes_0)
    var_1 = validation_2.__str__()
    var_2 = validation_2.to_try()
    var_3 = validation_2.__eq__(validation_1)
    var_3.to_lazy()


def test_case_1():
    str_0 = "dy5Yb]x=lD2v"
    tuple_0 = ()
    validation_0 = module_0.Validation(tuple_0, tuple_0)
    var_0 = validation_0.__eq__(str_0)
    var_0.to_try()


def test_case_2():
    tuple_0 = ()
    validation_0 = module_0.Validation(tuple_0, tuple_0)
    var_0 = validation_0.to_box()
    var_1 = validation_0.__str__()
    var_2 = var_0.to_try()
    validation_0.ap(var_2)


def test_case_3():
    object_0 = module_1.object()
    list_0 = []
    validation_0 = module_0.Validation(list_0, list_0)
    var_0 = validation_0.to_either()
    var_0.bind(object_0)


def test_case_4():
    int_0 = -298
    dict_0 = {int_0: int_0, int_0: int_0}
    validation_0 = module_0.Validation(int_0, dict_0)
    var_0 = validation_0.to_either()
    var_1 = validation_0.to_box()
    var_1.map(var_1)


def test_case_5():
    set_0 = set()
    validation_0 = module_0.Validation(set_0, set_0)
    var_0 = validation_0.to_try()
    var_1 = validation_0.to_maybe()
    validation_0.map(set_0)


def test_case_6():
    bool_0 = False
    validation_0 = module_0.Validation(bool_0, bool_0)


def test_case_7():
    int_0 = -1348
    validation_0 = module_0.Validation(int_0, int_0)
    validation_0.__str__()


def test_case_8():
    int_0 = 515
    bool_0 = False
    validation_0 = module_0.Validation(int_0, bool_0)
    validation_0.is_fail()


def test_case_9():
    bool_0 = False
    validation_0 = module_0.Validation(bool_0, bool_0)
    bytes_0 = b"\xd4%O\x9d\xa4\xa8\xeeJ\xc3o\x0e"
    validation_1 = module_0.Validation(bytes_0, bytes_0)
    var_0 = validation_1.to_maybe()
    validation_2 = module_0.Validation(bytes_0, bytes_0)
    var_1 = validation_2.__str__()
    var_2 = validation_2.to_try()
    var_3 = validation_2.__eq__(validation_1)
    var_4 = var_0.to_try()
    validation_2.map(validation_0)


def test_case_10():
    none_type_0 = None
    str_0 = "dy5Yb]x=lD2v"
    tuple_0 = ()
    validation_0 = module_0.Validation(tuple_0, tuple_0)
    var_0 = validation_0.__eq__(str_0)
    validation_0.bind(none_type_0)


def test_case_11():
    bool_0 = False
    validation_0 = module_0.Validation(bool_0, bool_0)
    bytes_0 = b"\xd4%O\x9d\xa4\xa8\xeeJ\xc3o\x0e"
    validation_1 = module_0.Validation(bytes_0, bytes_0)
    var_0 = validation_1.to_maybe()
    validation_2 = module_0.Validation(bytes_0, bytes_0)
    var_1 = validation_2.__str__()
    var_2 = validation_2.to_try()
    var_3 = validation_2.__eq__(validation_1)
    var_4 = var_0.to_try()
    validation_1.ap(var_4)


def test_case_12():
    tuple_0 = ()
    validation_0 = module_0.Validation(tuple_0, tuple_0)
    bool_0 = False
    validation_1 = module_0.Validation(bool_0, bool_0)
    var_0 = validation_1.to_box()
    var_0.to_box()


def test_case_13():
    bytes_0 = b"\x01\x179\x9e"
    validation_0 = module_0.Validation(bytes_0, bytes_0)
    var_0 = validation_0.to_lazy()
    var_1 = var_0.__str__()
    var_1.map(bytes_0)


def test_case_14():
    str_0 = "\n        :param value: value to store in Box\n        :type value: Any\n        "
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.to_try()
    var_0.is_fail()


def test_case_15():
    bool_0 = False
    validation_0 = module_0.Validation(bool_0, bool_0)
    var_0 = validation_0.to_lazy()
    var_1 = validation_0.to_lazy()
    var_2 = var_1.to_try()
    validation_0.map(validation_0)


def test_case_16():
    bool_0 = False
    validation_0 = module_0.Validation(bool_0, bool_0)
    bytes_0 = b"\xd4%O\x9d\xa4\xa8\xeeJ\xc3o\x0e"
    validation_1 = module_0.Validation(bytes_0, bytes_0)
    var_0 = validation_1.__eq__(validation_0)
    validation_2 = module_0.Validation(bytes_0, bytes_0)
    var_1 = validation_2.__str__()
    var_2 = validation_2.to_try()
    var_2.to_lazy()
