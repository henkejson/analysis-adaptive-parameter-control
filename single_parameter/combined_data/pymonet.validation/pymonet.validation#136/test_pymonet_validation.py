# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.validation as module_0


def test_case_0():
    str_0 = "\n        Transform Either to Try.\n\n        :returns: Lazy monad with function returning previous value\n        :rtype: Lazy[Function() -> A]\n        "
    validation_0 = module_0.Validation(str_0, str_0)
    none_type_0 = None
    var_0 = validation_0.to_maybe()
    validation_1 = module_0.Validation(validation_0, none_type_0)
    var_1 = validation_0.__eq__(none_type_0)
    var_1.to_box()


def test_case_1():
    str_0 = "Q-28:&`Y"
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.__str__()
    var_1 = validation_0.to_try()
    var_2 = validation_0.to_either()
    var_3 = var_2.bind(str_0)
    var_4 = var_3.to_maybe()
    var_5 = var_1.__str__()
    var_4.to_maybe()


def test_case_2():
    str_0 = "\n        :param semigroup: other semigroup to concat\n        :type semigroup: First[B]\n        :returns: new First with first value\n        :rtype: First[A]\n        "
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.to_either()
    var_1 = var_0.to_lazy()
    var_2 = var_1.to_box()
    var_2.is_fail()


def test_case_3():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)


def test_case_4():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    validation_0.to_try()


def test_case_5():
    str_0 = "\n        Transform Either to Try.\n\n        :returns: Lazy monad with function returning previous value\n        :rtype: Lazy[Function() -> A]\n        "
    validation_0 = module_0.Validation(str_0, str_0)
    none_type_0 = None
    var_0 = validation_0.to_maybe()
    var_1 = validation_0.__eq__(none_type_0)
    validation_1 = module_0.Validation(validation_0, none_type_0)
    var_2 = validation_0.__eq__(none_type_0)
    validation_1.is_fail()


def test_case_6():
    str_0 = "\n        :param semigroup: other semigroup to concat\n        :type semigroup: First[B]\n        :returns: new First with first value\n        :rtype: First[A]\n        "
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.to_either()
    var_1 = var_0.to_lazy()
    var_2 = var_1.to_box()
    validation_0.map(var_0)


def test_case_7():
    none_type_0 = None
    bytes_0 = b"\xa2\xe7'g\x87ezk3"
    validation_0 = module_0.Validation(bytes_0, bytes_0)
    validation_0.bind(none_type_0)


def test_case_8():
    none_type_0 = None
    bytes_0 = b"\xd0\xd4\x0e\xfa\xe6\xb4\xfdLs\xb1d\xf7\xa1\x10io\x97"
    none_type_1 = None
    validation_0 = module_0.Validation(bytes_0, none_type_1)
    validation_0.ap(none_type_0)


def test_case_9():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    var_0 = validation_0.to_box()
    validation_0.__str__()


def test_case_10():
    int_0 = 3646
    validation_0 = module_0.Validation(int_0, int_0)
    var_0 = validation_0.to_lazy()
    var_0.is_success()


def test_case_11():
    bool_0 = True
    validation_0 = module_0.Validation(bool_0, bool_0)
    var_0 = validation_0.to_lazy()
    var_1 = var_0.map(bool_0)
    var_1.to_box()


def test_case_12():
    bytes_0 = b"\x1b\x04\x13\xfe\x85\xf0\x01\x0cs\x99XQ\xcd\xf2\xff\xa5\xc8R"
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    validation_1 = module_0.Validation(bytes_0, bytes_0)
    var_0 = validation_0.__eq__(validation_1)
    var_1 = validation_1.__eq__(bytes_0)
    validation_1.ap(bytes_0)


def test_case_13():
    bytes_0 = b"\x80\xb9"
    dict_0 = {}
    validation_0 = module_0.Validation(dict_0, dict_0)
    var_0 = validation_0.__str__()
    var_0.bind(bytes_0)


def test_case_14():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    bool_0 = False
    var_0 = validation_0.__eq__(bool_0)
    var_1 = validation_0.__eq__(validation_0)
    validation_0.to_either()


def test_case_15():
    set_0 = set()
    validation_0 = module_0.Validation(set_0, set_0)
    var_0 = validation_0.to_either()
    str_0 = "[{\x0cuG\\Q)z>%G N\n"
    validation_1 = module_0.Validation(str_0, str_0)
    var_1 = validation_1.to_either()
    var_2 = var_1.to_lazy()
    var_3 = var_2.to_maybe()
    var_4 = var_3.to_lazy()
    var_5 = var_4.to_box()
    var_6 = var_5.__eq__(var_0)
    var_6.is_success()


def test_case_16():
    tuple_0 = ()
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, tuple_0)
    var_0 = validation_0.to_maybe()
    var_1 = var_0.__str__()
    validation_1 = module_0.Validation(tuple_0, tuple_0)
    var_2 = validation_1.is_success()
    var_3 = validation_1.to_lazy()
    var_2.is_fail()
