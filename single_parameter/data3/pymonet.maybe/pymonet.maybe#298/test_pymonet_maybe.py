# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.maybe as module_0


def test_case_0():
    bytes_0 = b"\x95\x9a\x8f\xda\xffF="
    maybe_0 = module_0.Maybe(bytes_0, bytes_0)


def test_case_1():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)


def test_case_2():
    none_type_0 = None
    bool_0 = False
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    bool_1 = maybe_0.__eq__(maybe_0)
    var_0 = maybe_0.to_either()
    var_0.ap(none_type_0)


def test_case_3():
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.to_try()
    var_1 = maybe_0.filter(var_0)
    var_2 = maybe_0.to_try()
    bool_1 = var_2.__eq__(none_type_0)
    maybe_1 = module_0.Maybe(none_type_0, bool_0)
    var_3 = maybe_0.filter(maybe_0)
    bool_2 = maybe_1.__eq__(bool_0)
    var_4 = maybe_0.get_or_else(var_3)
    var_5 = var_4.to_validation()
    var_5.map(var_3)


def test_case_4():
    complex_0 = -1586 - 2678.583j
    bool_0 = False
    bytes_0 = b"\xb2\x8c\x16\x02\x94\xeb\xcf\x90\x1c"
    dict_0 = {bytes_0: bytes_0, bool_0: bool_0}
    tuple_0 = (bool_0, bool_0, bytes_0, dict_0)
    none_type_0 = None
    bool_1 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_1)
    var_0 = maybe_0.map(tuple_0)
    var_1 = var_0.bind(complex_0)
    none_type_1 = None
    bool_2 = True
    maybe_1 = module_0.Maybe(none_type_1, bool_2)
    var_2 = maybe_1.to_validation()


def test_case_5():
    bool_0 = False
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    none_type_0 = None
    var_0 = maybe_0.get_or_else(none_type_0)
    var_1 = maybe_0.to_validation()
    maybe_0.map(var_1)


def test_case_6():
    tuple_0 = ()
    str_0 = "\x0c);"
    bool_0 = True
    tuple_1 = (str_0, bool_0)
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.to_try()
    maybe_1 = module_0.Maybe(tuple_1, bool_0)
    var_1 = maybe_1.ap(tuple_0)
    var_2 = var_1.to_validation()
    var_3 = var_2.to_lazy()
    var_4 = maybe_1.to_either()
    var_5 = var_1.bind(var_4)
    bool_1 = var_4.__eq__(maybe_0)


def test_case_7():
    bool_0 = True
    bool_1 = False
    maybe_0 = module_0.Maybe(bool_0, bool_1)
    var_0 = maybe_0.to_lazy()
    maybe_0.bind(maybe_0)


def test_case_8():
    dict_0 = {}
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    maybe_0.ap(dict_0)


def test_case_9():
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, none_type_0)
    var_0 = maybe_0.to_try()
    bool_1 = var_0.__eq__(bool_0)
    maybe_1 = module_0.Maybe(none_type_0, bool_0)
    var_1 = maybe_1.to_either()
    maybe_0.filter(var_1)


def test_case_10():
    bool_0 = True
    none_type_0 = None
    tuple_0 = ()
    bool_1 = True
    maybe_0 = module_0.Maybe(tuple_0, bool_1)
    var_0 = maybe_0.to_either()
    var_1 = var_0.ap(none_type_0)
    var_1.filter(bool_0)


def test_case_11():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.to_either()
    none_type_1 = None
    str_0 = "\nUvb\n`{+9M@n"
    none_type_2 = None
    var_1 = var_0.to_lazy()
    maybe_1 = module_0.Maybe(str_0, none_type_2)
    maybe_1.map(none_type_1)


def test_case_12():
    bytes_0 = b"\xd2"
    int_0 = 1431
    maybe_0 = module_0.Maybe(int_0, int_0)
    var_0 = maybe_0.to_box()
    var_0.ap(bytes_0)


def test_case_13():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.to_box()
    var_0.to_box()


def test_case_14():
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.to_try()
    var_1 = maybe_0.filter(var_0)
    var_2 = maybe_0.to_try()
    var_3 = maybe_0.to_lazy()
    bool_1 = var_2.__eq__(none_type_0)
    maybe_1 = module_0.Maybe(none_type_0, bool_0)
    var_4 = maybe_0.filter(maybe_0)
    bool_2 = maybe_1.__eq__(bool_0)
    var_5 = maybe_0.get_or_else(var_4)
    var_6 = var_5.to_validation()
    var_6.map(var_4)


def test_case_15():
    float_0 = 80.6233
    list_0 = []
    bool_0 = False
    maybe_0 = module_0.Maybe(list_0, bool_0)
    var_0 = maybe_0.to_lazy()
    var_1 = var_0.ap(float_0)
    var_1.filter(maybe_0)


def test_case_16():
    float_0 = 82.09
    maybe_0 = module_0.Maybe(float_0, float_0)
    var_0 = maybe_0.to_try()
    var_0.to_lazy()


def test_case_17():
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, none_type_0)
    var_0 = maybe_0.to_try()
    bool_1 = var_0.__eq__(bool_0)
    maybe_1 = module_0.Maybe(none_type_0, bool_0)
    var_1 = maybe_1.to_either()
    var_2 = maybe_1.to_validation()
    var_3 = var_1.to_box()
    var_2.to_validation()


def test_case_18():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.to_lazy()
    var_1 = maybe_0.to_box()
    var_2 = var_0.to_either()
    var_3 = maybe_0.to_lazy()
    var_2.filter(var_0)


def test_case_19():
    none_type_0 = None
    str_0 = "FP8;W_qPW$*1-L\n"
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.filter(none_type_0)
    var_1 = var_0.ap(str_0)
    var_2 = var_0.filter(var_1)
    var_3 = var_1.map(var_1)
    var_4 = var_0.map(none_type_0)
    bool_1 = True
    maybe_1 = module_0.Maybe(str_0, bool_1)
    bool_2 = maybe_1.__eq__(var_3)
    var_5 = var_2.to_box()
    var_6 = var_4.to_validation()
    var_7 = var_1.to_lazy()
    var_8 = maybe_0.to_lazy()


def test_case_20():
    int_0 = 1
    maybe_0 = module_0.Maybe(int_0, int_0)
    set_0 = set()
    none_type_0 = None
    var_0 = maybe_0.ap(set_0)
    var_1 = maybe_0.bind(var_0)
    str_0 = "FP8;W_qPW$*1-L\n"
    bool_0 = True
    maybe_1 = module_0.Maybe(none_type_0, bool_0)
    var_2 = maybe_1.filter(none_type_0)
    var_3 = var_2.to_either()
    var_4 = var_2.ap(str_0)
    var_5 = var_2.filter(var_4)
    var_6 = var_4.map(var_3)
    var_7 = var_2.map(none_type_0)
    bool_1 = True
    maybe_2 = module_0.Maybe(str_0, bool_1)
    var_8 = var_0.to_box()
    bool_2 = maybe_2.__eq__(var_6)
    var_9 = maybe_2.ap(none_type_0)
    var_10 = maybe_0.map(maybe_2)
    bool_3 = False
    maybe_3 = module_0.Maybe(maybe_0, bool_3)
    var_11 = var_6.filter(var_9)
    var_12 = maybe_2.map(var_0)
    bool_4 = True
    maybe_4 = module_0.Maybe(var_3, bool_2)
    var_13 = var_0.to_either()
    var_14 = var_4.to_lazy()
    maybe_5 = module_0.Maybe(bool_4, maybe_1)
    var_15 = var_14.to_validation()
    var_16 = var_15.to_lazy()
    bool_5 = var_2.__eq__(maybe_2)


def test_case_21():
    int_0 = 1
    maybe_0 = module_0.Maybe(int_0, int_0)
    set_0 = set()
    none_type_0 = None
    var_0 = maybe_0.ap(set_0)
    var_1 = maybe_0.bind(var_0)
    str_0 = "FP8;W_qPW$*1-L\n"
    bool_0 = True
    maybe_1 = module_0.Maybe(var_1, maybe_0)
    var_2 = maybe_1.filter(none_type_0)
    var_3 = maybe_0.to_either()
    var_4 = var_3.ap(var_1)
    var_5 = var_2.filter(var_0)
    var_6 = var_3.map(bool_0)
    var_7 = var_3.map(str_0)
    maybe_2 = module_0.Maybe(set_0, int_0)
    var_8 = maybe_0.to_box()
    bool_1 = maybe_1.__eq__(var_1)
    var_9 = var_0.ap(maybe_2)
    var_10 = var_3.map(var_3)
    maybe_3 = module_0.Maybe(int_0, int_0)
    var_11 = var_1.filter(var_0)
    var_12 = maybe_3.map(maybe_0)
    var_3.to_either()
