# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.maybe as module_0


def test_case_0():
    str_0 = "7R|&%K\\q\x0b\x0c|0k6l"
    maybe_0 = module_0.Maybe(str_0, str_0)


def test_case_1():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)


def test_case_2():
    none_type_0 = None
    set_0 = set()
    bool_0 = True
    maybe_0 = module_0.Maybe(set_0, bool_0)
    var_0 = maybe_0.filter(none_type_0)
    var_1 = var_0.ap(set_0)
    bool_1 = maybe_0.__eq__(var_1)
    var_2 = var_0.to_either()
    var_2.to_either()


def test_case_3():
    none_type_0 = None
    bool_0 = True
    bool_1 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_1)
    bool_2 = maybe_0.__eq__(bool_0)


def test_case_4():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.filter(bool_0)
    str_0 = 'v7\x0cwW$\tJ*I1"1'
    bool_1 = False
    var_1 = maybe_0.map(bool_1)
    maybe_1 = module_0.Maybe(str_0, bool_1)
    var_2 = maybe_0.get_or_else(var_1)
    var_3 = var_0.to_try()
    maybe_1.filter(var_1)


def test_case_5():
    str_0 = "\n    The Try control gives us the ability write safe code\n    without focusing on try-catch blocks in the presence of exceptions.\n    "
    bool_0 = False
    maybe_0 = module_0.Maybe(str_0, bool_0)
    var_0 = maybe_0.to_lazy()
    var_1 = var_0.to_try()
    none_type_0 = None
    maybe_1 = module_0.Maybe(none_type_0, none_type_0)
    maybe_1.map(var_1)


def test_case_6():
    float_0 = -3201.52625
    bool_0 = True
    maybe_0 = module_0.Maybe(float_0, bool_0)
    var_0 = maybe_0.to_either()
    var_1 = maybe_0.to_box()
    bool_1 = True
    var_2 = maybe_0.bind(bool_0)
    var_3 = var_0.map(var_2)
    bytes_0 = b"J[\x8f\x11\xb9\x9e\xad0>\xd3\xac\x8f\x11f\xb0|1\xa3"
    bool_2 = var_1.__eq__(bytes_0)
    bool_3 = var_2.__eq__(bool_1)
    var_4 = maybe_0.to_lazy()
    var_5 = var_2.ap(float_0)
    bool_4 = var_1.__eq__(var_5)
    var_6 = var_3.bind(bool_1)
    var_7 = var_6.to_validation()
    maybe_1 = module_0.Maybe(float_0, bool_1)


def test_case_7():
    bool_0 = True
    bool_1 = False
    maybe_0 = module_0.Maybe(bool_0, bool_1)
    maybe_0.bind(bool_0)


def test_case_8():
    int_0 = -3682
    bool_0 = True
    tuple_0 = (bool_0,)
    maybe_0 = module_0.Maybe(tuple_0, bool_0)
    var_0 = maybe_0.to_validation()
    var_1 = var_0.to_lazy()
    var_2 = var_1.bind(int_0)
    bool_1 = True
    maybe_1 = module_0.Maybe(bool_1, bool_1)
    none_type_0 = None
    var_3 = maybe_1.to_try()
    var_4 = maybe_1.filter(var_3)
    var_5 = var_4.ap(none_type_0)
    var_6 = maybe_1.map(var_5)


def test_case_9():
    bool_0 = False
    bool_1 = False
    dict_0 = {bool_1: bool_1, bool_1: bool_1, bool_1: bool_1, bool_1: bool_1}
    maybe_0 = module_0.Maybe(dict_0, bool_1)
    maybe_0.ap(bool_0)


def test_case_10():
    int_0 = -257
    maybe_0 = module_0.Maybe(int_0, int_0)
    var_0 = maybe_0.filter(maybe_0)
    var_1 = var_0.to_validation()
    var_2 = maybe_0.filter(var_1)
    int_1 = 0
    bool_0 = False
    maybe_1 = module_0.Maybe(int_1, bool_0)


def test_case_11():
    bool_0 = True
    set_0 = {bool_0, bool_0, bool_0}
    bool_1 = True
    bool_2 = False
    maybe_0 = module_0.Maybe(bool_1, bool_2)
    maybe_0.filter(set_0)


def test_case_12():
    int_0 = -717
    none_type_0 = None
    maybe_0 = module_0.Maybe(int_0, none_type_0)
    var_0 = maybe_0.to_try()
    int_1 = -133
    none_type_1 = None
    var_1 = maybe_0.to_validation()
    maybe_1 = module_0.Maybe(none_type_1, none_type_1)
    var_2 = maybe_1.get_or_else(none_type_1)
    var_3 = maybe_1.to_validation()
    var_4 = var_3.to_either()
    var_5 = maybe_1.to_validation()
    var_6 = maybe_1.to_try()
    var_7 = maybe_1.to_either()
    var_8 = var_5.to_box()
    var_9 = var_3.to_try()
    var_10 = var_7.to_box()
    var_11 = var_5.to_either()
    var_10.map(int_1)


def test_case_13():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.filter(bool_0)
    str_0 = "iR]Dr8-wuuTc\x0b1LUQLtI"
    bool_1 = False
    var_1 = var_0.to_either()
    var_2 = maybe_0.to_box()
    maybe_1 = module_0.Maybe(str_0, bool_1)
    var_3 = var_0.to_try()
    var_4 = var_0.to_try()
    maybe_1.filter(var_2)


def test_case_14():
    str_0 = "\n        Transform Box into successfull Try.\n\n        :returns: successfull Try monad with previous value\n        :rtype: Try[A]\n        "
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.to_either()
    var_0.get_or_else(str_0)


def test_case_15():
    str_0 = 'ci^"\x0cw-iJqdOn9Kj'
    float_0 = 986.201599
    maybe_0 = module_0.Maybe(float_0, float_0)
    var_0 = maybe_0.to_box()
    var_0.get_or_else(str_0)


def test_case_16():
    bytes_0 = b"@\xa9\xfdw\xdf?\x02\xf37\xd9\x12"
    bool_0 = False
    maybe_0 = module_0.Maybe(bytes_0, bool_0)
    var_0 = maybe_0.to_box()
    var_1 = var_0.to_lazy()
    bool_1 = True
    maybe_1 = module_0.Maybe(bytes_0, bool_1)


def test_case_17():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    int_0 = 1454
    dict_0 = {int_0: int_0}
    maybe_1 = module_0.Maybe(dict_0, int_0)
    var_0 = maybe_1.to_lazy()
    var_0.to_lazy()


def test_case_18():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.to_try()
    maybe_0.filter(var_0)


def test_case_19():
    bool_0 = True
    list_0 = [bool_0, bool_0]
    str_0 = ""
    maybe_0 = module_0.Maybe(str_0, str_0)
    var_0 = maybe_0.to_validation()
    var_1 = var_0.to_box()
    var_1.filter(list_0)


def test_case_20():
    str_0 = "Mj"
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(str_0, bool_0)
    var_0 = maybe_0.to_lazy()
    var_1 = maybe_0.bind(none_type_0)
    maybe_1 = var_0.to_try()
    maybe_1.ap(var_0)


def test_case_21():
    bool_0 = True
    none_type_0 = None
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.to_validation()
    maybe_1 = module_0.Maybe(none_type_0, var_0)
    var_1 = maybe_0.filter(var_0)
    var_2 = maybe_1.ap(bool_0)
    var_3 = var_1.bind(var_0)
    var_4 = var_3.filter(var_0)
    var_5 = var_1.to_validation()
    var_6 = maybe_0.bind(none_type_0)
    str_0 = "iR]Dr8-wuuTc\x0b1LUQLtI"
    bool_1 = True
    maybe_2 = module_0.Maybe(str_0, bool_1)
    var_7 = var_1.to_try()
    bool_2 = maybe_2.__eq__(var_5)
    var_8 = maybe_1.to_lazy()
    var_9 = maybe_2.bind(var_6)
    var_10 = var_9.ap(var_5)
    maybe_3 = module_0.Maybe(none_type_0, var_8)
    bool_3 = maybe_3.__eq__(var_6)
    var_5.bind(var_2)


def test_case_22():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    bool_0 = maybe_0.__eq__(maybe_0)
    var_0 = maybe_0.to_lazy()
    maybe_1 = module_0.Maybe(var_0, bool_0)
    var_1 = maybe_0.to_validation()
    var_2 = maybe_1.to_box()
    var_0.to_lazy()
