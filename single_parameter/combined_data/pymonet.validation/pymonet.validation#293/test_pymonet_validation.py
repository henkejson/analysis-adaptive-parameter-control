# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.validation as module_0


def test_case_0():
    bool_0 = False
    validation_0 = module_0.Validation(bool_0, bool_0)
    var_0 = validation_0.__eq__(bool_0)
    var_0.to_box()


def test_case_1():
    tuple_0 = ()
    validation_0 = module_0.Validation(tuple_0, tuple_0)
    var_0 = validation_0.__str__()
    var_0.to_either()


def test_case_2():
    str_0 = "p^I]g#C|hT_lG<"
    validation_0 = module_0.Validation(str_0, str_0)
    bool_0 = True
    var_0 = validation_0.__str__()
    validation_1 = module_0.Validation(bool_0, bool_0)
    var_1 = validation_1.__eq__(bool_0)
    var_1.ap(bool_0)


def test_case_3():
    complex_0 = -1110.5 - 2135.83199j
    bytes_0 = b"\x0e\xe1\xe7\x82qu\xceI"
    set_0 = {bytes_0}
    validation_0 = module_0.Validation(set_0, set_0)
    var_0 = validation_0.is_success()
    var_1 = validation_0.to_either()
    var_2 = validation_0.to_either()
    var_3 = var_1.__eq__(complex_0)
    var_3.to_lazy()


def test_case_4():
    str_0 = "2ig`"
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.to_maybe()
    var_1 = var_0.to_either()
    var_2 = var_1.to_lazy()
    var_2.is_fail()


def test_case_5():
    bytes_0 = b"AWN\x1bQ\xe8\x9f\x0cy\xfb\xaeyz\xc9"
    bytes_0.is_success()


def test_case_6():
    bytes_0 = b"AWN\x1bQ\xe8\x9f\x0cy\xfb\xaeyz\xc9"
    var_0 = module_0.Validation(bytes_0, bytes_0)


def test_case_7():
    int_0 = 531
    validation_0 = module_0.Validation(int_0, int_0)
    validation_0.__str__()


def test_case_8():
    bytes_0 = b"AWN\x1bQ\xe8\x9f\x0cy\xfb\xaeyz\xc9"
    validation_0 = module_0.Validation(bytes_0, bytes_0)
    var_0 = validation_0.is_success()
    var_1 = validation_0.to_lazy()
    var_2 = validation_0.is_fail()
    var_1.is_fail()


def test_case_9():
    set_0 = set()
    validation_0 = module_0.Validation(set_0, set_0)
    var_0 = validation_0.to_maybe()
    str_0 = "\"^E/23Vc\rtt4;Fpt'"
    validation_0.map(str_0)


def test_case_10():
    bool_0 = False
    bool_1 = True
    none_type_0 = None
    validation_0 = module_0.Validation(bool_1, none_type_0)
    validation_0.bind(bool_0)


def test_case_11():
    bool_0 = True
    validation_0 = module_0.Validation(bool_0, bool_0)
    none_type_0 = None
    validation_0.ap(none_type_0)


def test_case_12():
    int_0 = 531
    validation_0 = module_0.Validation(int_0, int_0)
    var_0 = validation_0.to_box()


def test_case_13():
    int_0 = 1527
    validation_0 = module_0.Validation(int_0, int_0)
    validation_1 = module_0.Validation(int_0, int_0)
    var_0 = validation_1.__eq__(validation_0)
    var_1 = validation_1.to_lazy()
    var_2 = var_1.to_either()
    validation_2 = module_0.Validation(int_0, int_0)
    validation_1.to_maybe()


def test_case_14():
    bool_0 = False
    bytes_0 = b"\x87\x18\xb30\xa8FB\xcd&\x02\xb6\xb4\xa0Q\xdd\xda\xdd\xf1\x06:"
    validation_0 = module_0.Validation(bool_0, bytes_0)
    var_0 = validation_0.to_maybe()
    validation_1 = module_0.Validation(bool_0, bytes_0)
    var_1 = validation_1.to_try()
    var_0.to_maybe()


def test_case_15():
    int_0 = 1527
    validation_0 = module_0.Validation(int_0, int_0)
    validation_1 = module_0.Validation(int_0, int_0)
    var_0 = validation_1.__eq__(validation_0)
    var_1 = validation_1.to_lazy()
    validation_2 = module_0.Validation(int_0, int_0)
    validation_1.to_maybe()


def test_case_16():
    tuple_0 = ()
    validation_0 = module_0.Validation(tuple_0, tuple_0)
    var_0 = validation_0.to_either()
    var_1 = validation_0.to_box()
    var_2 = validation_0.to_either()
    var_3 = var_2.to_box()
    var_2.is_fail()


def test_case_17():
    int_0 = 1527
    none_type_0 = None
    validation_0 = module_0.Validation(int_0, none_type_0)
    validation_1 = module_0.Validation(int_0, int_0)
    var_0 = validation_1.__eq__(validation_0)
    var_1 = validation_1.to_lazy()
    var_2 = var_1.to_either()
    validation_1.to_maybe()
