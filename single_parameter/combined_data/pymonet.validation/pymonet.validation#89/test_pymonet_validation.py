# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.validation as module_0
import builtins as module_1


def test_case_0():
    str_0 = "\n        Transform Either to Try.\n\n        :returns: resolved Try monad with previous value. Right is resolved successfully, Left not.\n        :rtype: Box[A]\n        "
    int_0 = 434
    validation_0 = module_0.Validation(int_0, int_0)
    var_0 = validation_0.__eq__(str_0)
    var_0.to_try()


def test_case_1():
    set_0 = set()
    list_0 = [set_0]
    validation_0 = module_0.Validation(list_0, list_0)
    var_0 = validation_0.__str__()


def test_case_2():
    set_0 = set()
    validation_0 = module_0.Validation(set_0, set_0)
    var_0 = validation_0.to_maybe()
    var_1 = var_0.to_either()
    var_2 = var_1.to_try()
    var_2.to_box()


def test_case_3():
    none_type_0 = None
    none_type_0.to_try()


def test_case_4():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)


def test_case_5():
    str_0 = "lA2V K|E?"
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.is_fail()


def test_case_6():
    str_0 = "\n    Max is a Monoid that will combines 2 numbers, resulting in the largest of the two.\n    "
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    validation_0.map(str_0)


def test_case_7():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)
    validation_0.bind(none_type_0)


def test_case_8():
    none_type_0 = None
    none_type_1 = None
    validation_0 = module_0.Validation(none_type_1, none_type_1)
    validation_0.ap(none_type_0)


def test_case_9():
    set_0 = set()
    list_0 = [set_0]
    validation_0 = module_0.Validation(list_0, list_0)
    var_0 = validation_0.__str__()
    var_1 = validation_0.to_box()
    var_0.to_either()


def test_case_10():
    object_0 = module_1.object()
    none_type_0 = None
    bool_0 = False
    validation_0 = module_0.Validation(bool_0, bool_0)
    var_0 = validation_0.to_lazy()
    validation_1 = module_0.Validation(object_0, none_type_0)


def test_case_11():
    bool_0 = True
    bool_1 = True
    str_0 = ':P~Qe+Y\na"'
    list_0 = [str_0, str_0, str_0]
    list_1 = [list_0]
    validation_0 = module_0.Validation(bool_1, list_0)
    var_0 = validation_0.to_either()
    validation_1 = module_0.Validation(list_1, list_1)
    var_1 = validation_1.__str__()
    var_2 = validation_1.__eq__(var_1)
    validation_2 = validation_1.to_lazy()
    var_3 = validation_2.bind(bool_0)
    var_4 = validation_2.to_either()
    var_5 = validation_2.to_try()


def test_case_12():
    bytes_0 = b"\xcdp\x80\xca\xea["
    validation_0 = module_0.Validation(bytes_0, bytes_0)
    var_0 = validation_0.to_lazy()
    int_0 = 434
    none_type_0 = None
    validation_1 = module_0.Validation(none_type_0, int_0)
    var_1 = validation_0.__eq__(int_0)
    validation_1.to_try()


def test_case_13():
    set_0 = set()
    list_0 = [set_0, set_0]
    validation_0 = module_0.Validation(list_0, list_0)
    var_0 = validation_0.__eq__(validation_0)
    var_1 = validation_0.to_maybe()
    var_0.to_either()


def test_case_14():
    set_0 = set()
    list_0 = [set_0]
    validation_0 = module_0.Validation(list_0, list_0)
    var_0 = validation_0.__eq__(validation_0)
    var_1 = validation_0.__str__()
    var_2 = validation_0.to_maybe()
    var_3 = validation_0.to_lazy()
    var_4 = validation_0.to_either()


def test_case_15():
    set_0 = set()
    validation_0 = module_0.Validation(set_0, set_0)
    var_0 = validation_0.to_either()
    var_1 = validation_0.to_lazy()
    var_2 = validation_0.to_lazy()
    var_3 = var_1.to_maybe()
    validation_1 = module_0.Validation(set_0, set_0)
    var_4 = validation_1.to_box()
    validation_1.ap(set_0)


def test_case_16():
    set_0 = set()
    list_0 = [set_0]
    validation_0 = module_0.Validation(list_0, list_0)
    var_0 = validation_0.__eq__(validation_0)
    validation_1 = module_0.Validation(validation_0, set_0)
    var_1 = validation_0.__eq__(validation_1)
    var_2 = validation_1.__str__()
    var_1.to_maybe()
