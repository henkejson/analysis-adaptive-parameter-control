# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.validation as module_0


def test_case_0():
    str_0 = "_iP[1 >(@\x0cLE`\x0c-"
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.__eq__(str_0)
    validation_1 = module_0.Validation(str_0, str_0)
    var_1 = validation_0.to_maybe()


def test_case_1():
    str_0 = "> >mt"
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.__str__()


def test_case_2():
    str_0 = "?.94)d?w30T#a`W"
    tuple_0 = (str_0,)
    validation_0 = module_0.Validation(tuple_0, tuple_0)
    var_0 = validation_0.__str__()
    var_1 = validation_0.to_either()
    var_2 = validation_0.to_either()
    var_3 = var_2.__str__()
    var_3.to_either()


def test_case_3():
    dict_0 = {}
    validation_0 = module_0.Validation(dict_0, dict_0)
    var_0 = validation_0.to_maybe()
    var_0.is_success()


def test_case_4():
    bool_0 = False
    validation_0 = module_0.Validation(bool_0, bool_0)


def test_case_5():
    int_0 = -1360
    validation_0 = module_0.Validation(int_0, int_0)
    validation_0.to_try()


def test_case_6():
    str_0 = "TTo0t(/y3+zH$?"
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.is_fail()


def test_case_7():
    bool_0 = True
    str_0 = 'K[#dv\\`v=ii;@\x0b\nHx\x0b"'
    list_0 = [str_0, str_0, str_0, str_0]
    none_type_0 = None
    validation_0 = module_0.Validation(list_0, none_type_0)
    validation_0.map(bool_0)


def test_case_8():
    int_0 = 1062
    validation_0 = module_0.Validation(int_0, int_0)
    validation_0.bind(validation_0)


def test_case_9():
    int_0 = -1360
    none_type_0 = None
    validation_0 = module_0.Validation(int_0, int_0)
    validation_0.ap(none_type_0)


def test_case_10():
    str_0 = "_iP[H >(@\x0cLE`\x0c-"
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.to_box()


def test_case_11():
    int_0 = -3284
    validation_0 = module_0.Validation(int_0, int_0)
    var_0 = validation_0.to_lazy()


def test_case_12():
    bytes_0 = b"\xfe\x0f\xca\x1d\x96C/\xdf\x81)|\xc7"
    validation_0 = module_0.Validation(bytes_0, bytes_0)
    var_0 = validation_0.to_lazy()
    var_1 = validation_0.is_fail()
    var_2 = validation_0.__eq__(var_1)
    var_3 = var_0.ap(var_2)
    validation_1 = module_0.Validation(var_1, var_0)
    var_4 = var_1.__eq__(validation_0)
    validation_2 = module_0.Validation(var_1, var_0)
    var_5 = var_0.to_maybe()
    var_1.to_try()


def test_case_13():
    str_0 = "XogT0n"
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, str_0)
    var_0 = validation_0.to_maybe()


def test_case_14():
    bytes_0 = b"\xfe\x0f\xca\x1d\x96C/\xdf\x81)|\xc7"
    validation_0 = module_0.Validation(bytes_0, bytes_0)
    var_0 = validation_0.to_lazy()
    var_1 = validation_0.is_fail()
    var_2 = validation_0.__eq__(var_1)
    var_3 = var_1.__eq__(validation_0)
    validation_1 = module_0.Validation(bytes_0, bytes_0)
    var_4 = validation_1.__eq__(validation_0)
    var_4.to_lazy()


def test_case_15():
    bytes_0 = b"\xfe\x0f\xca\x1d\x96C/\xdf\x81)|\xc7"
    validation_0 = module_0.Validation(bytes_0, bytes_0)
    var_0 = validation_0.to_lazy()
    var_1 = validation_0.to_maybe()
    var_2 = validation_0.__eq__(var_1)
    validation_1 = module_0.Validation(var_1, var_0)
    var_3 = var_1.__eq__(validation_0)
    var_4 = validation_0.__str__()
    var_5 = validation_0.is_success()
    none_type_0 = None
    var_6 = validation_0.__eq__(validation_1)
    none_type_0.to_lazy()


def test_case_16():
    list_0 = []
    validation_0 = module_0.Validation(list_0, list_0)
    var_0 = validation_0.to_either()
    var_0.map(list_0)


def test_case_17():
    str_0 = ""
    validation_0 = module_0.Validation(str_0, str_0)
    var_0 = validation_0.__str__()
