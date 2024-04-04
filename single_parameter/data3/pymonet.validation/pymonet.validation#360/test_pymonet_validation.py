# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.validation as module_0


def test_case_0():
    str_0 = "\n        Transform Validation to Either.\n\n        :returns: Right monad with previous value when Validation has no errors, in other case Left with errors list\n        :rtype: Right[A] | Left[E]\n        "
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.__eq__(validation_0)
    var_0.to_lazy()


def test_case_1():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    var_0 = validation_0.__eq__(none_type_0)
    validation_1 = module_0.Validation(none_type_0, none_type_0)
    var_1 = validation_1.__eq__(validation_0)
    bytes_0 = b"\x9an\x95\xd5\xd2\x80fprY\x89\xdfz\xac\xe5\x95pc`\x97"
    var_2 = validation_1.__eq__(bytes_0)
    var_1.to_either()


def test_case_2():
    str_0 = "\n        Transform Validation to Either.\n\n        :returns: Right monad with previous value when Validation has no errors, in other case Left with errors list\n        :rtype: Right[A] | Left[E]\n        "
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.__eq__(validation_0)
    var_1 = validation_0.to_maybe()
    validation_1 = module_0.Validation(str_0, validation_0)
    var_2 = var_1.to_either()
    validation_2 = module_0.Validation(var_0, var_1)
    var_3 = validation_0.to_either()
    var_4 = var_0.__str__()
    var_4.map(validation_1)


def test_case_3():
    str_0 = "\n        Transform Validation to Either.\n\n        :returns: Right monad with previous value when Validation has no errors, in other case Left with errors list\n        :rtype: Right[A] | Left[E]\n        "
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.to_maybe()
    var_1 = validation_0.__eq__(validation_0)


def test_case_4():
    bool_0 = False
    validation_0 = module_0.Validation(bool_0, bool_0)


def test_case_5():
    str_0 = "#\r1N8B]F_3&?"
    list_0 = [str_0, str_0]
    validation_0 = module_0.Validation(list_0, list_0)
    var_0 = validation_0.is_success()


def test_case_6():
    tuple_0 = ()
    bool_0 = True
    tuple_1 = (tuple_0, bool_0)
    validation_0 = module_0.Validation(tuple_1, tuple_0)
    var_0 = validation_0.is_success()
    var_1 = validation_0.__eq__(var_0)
    validation_1 = module_0.Validation(tuple_0, bool_0)
    validation_2 = module_0.Validation(tuple_1, tuple_1)
    var_2 = validation_2.to_either()
    var_3 = validation_2.to_either()
    var_4 = validation_2.is_fail()
    var_0.ap(var_0)


def test_case_7():
    bytes_0 = b""
    validation_0 = module_0.Validation(bytes_0, bytes_0)
    var_0 = validation_0.to_box()
    none_type_0 = None
    validation_0.map(none_type_0)


def test_case_8():
    int_0 = 223
    int_1 = -971
    list_0 = [int_1, int_1]
    validation_0 = module_0.Validation(list_0, int_1)
    validation_0.bind(int_0)


def test_case_9():
    float_0 = -173.298311
    bool_0 = True
    dict_0 = {bool_0: bool_0, bool_0: bool_0, bool_0: bool_0, bool_0: bool_0}
    validation_0 = module_0.Validation(dict_0, dict_0)
    validation_0.ap(float_0)


def test_case_10():
    bool_0 = False
    none_type_0 = None
    validation_0 = module_0.Validation(bool_0, none_type_0)
    var_0 = validation_0.to_lazy()
    var_1 = var_0.to_either()
    var_1.is_fail()


def test_case_11():
    bool_0 = True
    str_0 = ""
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.to_try()
    var_1 = var_0.__eq__(bool_0)


def test_case_12():
    tuple_0 = ()
    bool_0 = False
    tuple_1 = (tuple_0, bool_0)
    validation_0 = module_0.Validation(tuple_1, tuple_0)
    var_0 = validation_0.is_success()
    var_1 = validation_0.__eq__(var_0)
    var_2 = validation_0.to_maybe()
    validation_1 = module_0.Validation(tuple_0, bool_0)
    validation_1.to_either()


def test_case_13():
    str_0 = "\n        Transform Validation to Either.\n\n        :returns: Right monad with previous value when Validation has no errors, in other case Left with errors list\n        :rtype: Right[A] | Left[E]\n        "
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.__str__()
    var_1 = validation_0.to_maybe()
    var_2 = validation_0.__eq__(validation_0)
    var_3 = var_2.__str__()
    var_4 = var_1.to_either()


def test_case_14():
    tuple_0 = ()
    bool_0 = True
    tuple_1 = (tuple_0, bool_0)
    validation_0 = module_0.Validation(tuple_1, tuple_0)
    var_0 = validation_0.is_success()
    var_1 = validation_0.__eq__(var_0)
    validation_1 = module_0.Validation(tuple_0, bool_0)
    validation_2 = module_0.Validation(tuple_1, tuple_1)
    var_2 = validation_2.to_either()
    var_3 = validation_1.__eq__(validation_2)
    var_3.to_either()


def test_case_15():
    tuple_0 = ()
    bool_0 = True
    tuple_1 = (tuple_0, bool_0)
    validation_0 = module_0.Validation(tuple_1, tuple_0)
    var_0 = validation_0.__eq__(tuple_0)
    validation_1 = module_0.Validation(tuple_0, bool_0)
    var_1 = validation_0.to_either()
    var_2 = bool_0.__str__()
    var_3 = var_2.__str__()
    var_3.map(validation_1)


def test_case_16():
    tuple_0 = ()
    bool_0 = True
    tuple_1 = (tuple_0, bool_0)
    validation_0 = module_0.Validation(tuple_1, tuple_0)
    var_0 = validation_0.is_success()
    validation_1 = module_0.Validation(tuple_0, bool_0)
    var_1 = validation_0.__str__()
    validation_1.map(validation_1)
