# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.validation as module_0
import builtins as module_1


def test_case_0():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    bytes_0 = b"^GS\xd5d\xdd\x8e*"
    validation_1 = module_0.Validation(bytes_0, bytes_0)
    var_0 = validation_1.__eq__(validation_0)
    var_0.is_fail()


def test_case_1():
    bytes_0 = b"\xe8\\:Ic\x08\x022\xa5\x8c"
    none_type_0 = None
    object_0 = module_1.object()
    dict_0 = {object_0: object_0, object_0: object_0, object_0: object_0}
    validation_0 = module_0.Validation(dict_0, object_0)
    var_0 = validation_0.__eq__(none_type_0)
    var_0.ap(bytes_0)


def test_case_2():
    str_0 = "6yebPR+(E9AA"
    list_0 = []
    validation_0 = module_0.Validation(list_0, list_0)
    var_0 = validation_0.__str__()
    var_0.map(str_0)


def test_case_3():
    bytes_0 = b"\xdb\x82m%J\x90m-\x1b"
    int_0 = 487
    validation_0 = module_0.Validation(int_0, bytes_0)
    var_0 = validation_0.to_either()
    var_1 = var_0.to_lazy()
    int_1 = 4850
    var_2 = var_0.__str__()
    var_3 = var_2.__str__()
    validation_1 = module_0.Validation(int_1, int_1)


def test_case_4():
    str_0 = "q{iv"
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.to_maybe()


def test_case_5():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)


def test_case_6():
    float_0 = -153.11326
    validation_0 = module_0.Validation(float_0, float_0)
    validation_0.is_fail()


def test_case_7():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    validation_0.map(none_type_0)


def test_case_8():
    str_0 = "\n        Transform Box into Validation.\n\n        :returns: successfull Validation monad with previous value\n        :rtype: Validation[A, []]\n        "
    validation_0 = module_0.Validation(str_0, str_0)
    validation_1 = module_0.Validation(validation_0, validation_0)
    validation_0.bind(str_0)


def test_case_9():
    str_0 = "\n        Transform Lazy into Try with constructor_fn result.\n        Try will be successful only when constructor_fn not raise anything.\n\n        :returns: Try with constructor_fn result\n        :rtype: Try[A] | Try[Error]\n        "
    set_0 = {str_0, str_0}
    validation_0 = module_0.Validation(str_0, set_0)
    var_0 = validation_0.to_either()
    none_type_0 = None
    var_1 = var_0.to_box()
    validation_1 = module_0.Validation(none_type_0, none_type_0)
    var_2 = validation_0.to_lazy()
    var_3 = validation_1.__eq__(validation_0)
    validation_0.ap(set_0)


def test_case_10():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    validation_1 = module_0.Validation(none_type_0, none_type_0)
    var_0 = validation_1.to_box()
    bytes_0 = b"\xa3a/t\xc9\x16\xe7\xa8\xa4@"
    var_0.map(bytes_0)


def test_case_11():
    str_0 = "\n        Transform Lazy into Try with constructor_fn result.\n        Try will be successful only when constructor_fn not raise anything.\n\n        :returns: Try with constructor_fn result\n        :rtype: Try[A] | Try[Error]\n        "
    set_0 = {str_0, str_0}
    validation_0 = module_0.Validation(str_0, set_0)
    var_0 = validation_0.to_either()
    tuple_0 = (var_0, var_0)
    validation_1 = module_0.Validation(var_0, var_0)
    var_1 = validation_0.to_lazy()
    var_2 = validation_1.__eq__(tuple_0)
    var_3 = var_2.__str__()
    var_4 = var_1.to_try()
    var_3.is_success()


def test_case_12():
    bool_0 = True
    str_0 = "'Jf;`K-R=&\\-!!ZPe"
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.is_fail()
    var_1 = var_0.__eq__(bool_0)
    var_2 = validation_0.to_try()
    var_1.to_box()


def test_case_13():
    str_0 = "\n        Transform Lazy into Try with constructor_fn result.\n        Try will be successful only when constructor_fn not raise anything.\n\n        :returns: Try with constructor_fn result\n        :rtype: Try[A] | Try[Error]\n        "
    set_0 = {str_0, str_0}
    validation_0 = module_0.Validation(str_0, set_0)
    var_0 = validation_0.to_either()
    var_1 = var_0.to_box()
    var_2 = var_0.to_lazy()
    var_3 = var_1.__eq__(var_1)
    var_4 = validation_0.__str__()
    var_5 = var_0.ap(var_3)
    var_6 = var_4.__str__()


def test_case_14():
    set_0 = set()
    validation_0 = module_0.Validation(set_0, set_0)
    var_0 = validation_0.to_maybe()
    validation_0.map(var_0)


def test_case_15():
    str_0 = "\n        Transform Lazy into Try with constructor_fn result.\n        Try will be successful only when constructor_fn not raise anything.\n\n        :returns: Try with constructor_fn result\n        :rtype: Try[A] | Try[Error]\n        "
    set_0 = {str_0, str_0}
    validation_0 = module_0.Validation(str_0, set_0)
    var_0 = validation_0.to_either()
    tuple_0 = (var_0, var_0)
    var_1 = var_0.to_box()
    var_2 = validation_0.to_lazy()
    var_3 = validation_0.__eq__(validation_0)
    var_4 = var_3.__str__()
    var_5 = var_2.ap(tuple_0)
    var_6 = var_1.__str__()


def test_case_16():
    set_0 = set()
    validation_0 = module_0.Validation(set_0, set_0)
    var_0 = validation_0.to_either()
