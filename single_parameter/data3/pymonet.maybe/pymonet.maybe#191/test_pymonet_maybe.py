# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.maybe as module_0


def test_case_0():
    bytes_0 = b"\xdf\xb1>\x1b\x1d\xffX\xbe^~xFS\xfdP"
    maybe_0 = module_0.Maybe(bytes_0, bytes_0)


def test_case_1():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)


def test_case_2():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.get_or_else(maybe_0)
    var_1 = maybe_0.filter(var_0)
    bool_1 = var_1.__eq__(maybe_0)
    var_2 = var_0.to_either()
    var_3 = var_1.to_either()
    var_4 = var_2.map(bool_0)


def test_case_3():
    bytes_0 = b"_\xa4D\xa7g\x01\x83"
    bool_0 = False
    maybe_0 = module_0.Maybe(bytes_0, bool_0)
    bool_1 = maybe_0.__eq__(bytes_0)
    list_0 = [bytes_0, bytes_0]
    str_0 = "$Gk"
    tuple_0 = (str_0,)
    maybe_1 = module_0.Maybe(tuple_0, str_0)
    bool_2 = False
    maybe_2 = module_0.Maybe(maybe_0, bool_2)
    var_0 = maybe_1.to_validation()
    var_1 = maybe_1.to_try()
    var_2 = var_1.map(list_0)


def test_case_4():
    bool_0 = True
    int_0 = -1920
    float_0 = 1925.3
    bool_1 = True
    maybe_0 = module_0.Maybe(float_0, bool_1)
    var_0 = maybe_0.map(int_0)
    var_1 = var_0.bind(bool_0)
    none_type_0 = None
    maybe_1 = module_0.Maybe(none_type_0, none_type_0)
    var_2 = maybe_1.to_try()
    var_3 = var_2.get_or_else(var_1)


def test_case_5():
    bytes_0 = b"_\xa4D\xa7g\x01\x83"
    bool_0 = False
    maybe_0 = module_0.Maybe(bytes_0, bool_0)
    bool_1 = maybe_0.__eq__(bytes_0)
    list_0 = [bytes_0, bytes_0]
    str_0 = "$Gk"
    tuple_0 = (str_0,)
    maybe_1 = module_0.Maybe(tuple_0, str_0)
    var_0 = maybe_1.ap(bool_0)
    var_1 = var_0.ap(list_0)
    var_2 = maybe_1.get_or_else(list_0)
    var_3 = var_0.to_validation()
    var_4 = var_3.to_either()
    maybe_0.map(var_1)


def test_case_6():
    complex_0 = -2336.89 + 733.1j
    none_type_0 = None
    bool_0 = False
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    maybe_0.bind(complex_0)


def test_case_7():
    int_0 = 1
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.to_either()
    var_1 = maybe_0.to_lazy()
    var_2 = var_1.to_either()
    var_3 = maybe_0.ap(none_type_0)
    bool_1 = maybe_0.__eq__(maybe_0)
    bool_2 = True
    maybe_1 = module_0.Maybe(int_0, bool_2)
    var_4 = maybe_1.to_try()


def test_case_8():
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.to_lazy()
    var_1 = var_0.to_either()
    var_2 = maybe_0.ap(bool_0)
    bool_1 = maybe_0.__eq__(var_2)
    var_3 = var_1.to_validation()
    bool_2 = False
    maybe_1 = module_0.Maybe(var_3, bool_2)
    maybe_1.ap(bool_1)


def test_case_9():
    bytes_0 = b"_\xa4D\xa7g\x01\x83"
    bool_0 = True
    maybe_0 = module_0.Maybe(bytes_0, bool_0)
    bool_1 = maybe_0.__eq__(bytes_0)
    str_0 = "$Gk"
    tuple_0 = (str_0,)
    maybe_1 = module_0.Maybe(maybe_0, bool_0)
    var_0 = maybe_1.to_either()
    var_1 = maybe_0.filter(var_0)
    var_2 = module_0.Maybe(tuple_0, bool_1)
    var_2.filter(var_0)


def test_case_10():
    bytes_0 = b"_\xa4D\xa7g\x01\x83"
    bool_0 = False
    maybe_0 = module_0.Maybe(bytes_0, bool_0)
    bool_1 = maybe_0.__eq__(bytes_0)
    str_0 = "$Gk"
    tuple_0 = (str_0,)
    maybe_1 = module_0.Maybe(tuple_0, str_0)
    bool_2 = False
    maybe_2 = module_0.Maybe(maybe_0, bool_2)
    var_0 = maybe_1.to_either()
    var_1 = maybe_1.filter(var_0)
    var_2 = maybe_0.get_or_else(maybe_1)
    var_2.filter(var_0)


def test_case_11():
    none_type_0 = None
    bool_0 = False
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    maybe_1 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_1.to_lazy()
    var_1 = maybe_1.to_either()
    var_2 = var_0.to_box()
    var_3 = maybe_1.to_try()
    var_4 = var_0.to_try()
    var_5 = maybe_1.to_validation()
    var_6 = maybe_1.to_either()
    var_7 = var_0.to_validation()
    bool_1 = var_5.__eq__(var_6)
    var_8 = var_5.to_try()
    var_3.to_try()


def test_case_12():
    none_type_0 = None
    bytes_0 = b"\x14\x7f\rn\xbdeK"
    maybe_0 = module_0.Maybe(bytes_0, bytes_0)
    var_0 = maybe_0.to_lazy()
    var_1 = maybe_0.to_box()
    var_2 = maybe_0.bind(var_0)
    maybe_1 = module_0.Maybe(var_0, maybe_0)
    var_3 = maybe_1.map(none_type_0)
    var_4 = var_2.to_either()
    maybe_2 = module_0.Maybe(var_3, maybe_1)
    bool_0 = True
    maybe_3 = module_0.Maybe(none_type_0, bool_0)
    var_5 = var_2.to_lazy()


def test_case_13():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.to_box()
    var_0.get_or_else(none_type_0)


def test_case_14():
    float_0 = -3231.26
    none_type_0 = None
    none_type_1 = None
    maybe_0 = module_0.Maybe(none_type_1, none_type_1)
    var_0 = maybe_0.to_lazy()
    var_1 = var_0.bind(none_type_0)
    var_1.get_or_else(float_0)


def test_case_15():
    bytes_0 = b"\x17\x8el\x1c&\xda\xc7\x8fr\x81"
    bool_0 = False
    maybe_0 = module_0.Maybe(bytes_0, bool_0)
    var_0 = maybe_0.to_lazy()
    int_0 = 1042
    bool_1 = True
    maybe_1 = module_0.Maybe(int_0, bool_1)
    var_1 = maybe_1.to_try()
    var_1.to_try()


def test_case_16():
    bool_0 = True
    bool_1 = False
    maybe_0 = module_0.Maybe(bool_0, bool_1)
    var_0 = maybe_0.to_lazy()
    var_1 = var_0.to_box()
    var_2 = var_1.to_try()
    var_2.to_validation()


def test_case_17():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    bool_0 = True
    maybe_1 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_1.bind(none_type_0)
    var_1 = maybe_1.get_or_else(maybe_1)
    var_2 = maybe_1.filter(none_type_0)
    var_3 = var_0.to_lazy()
    bool_1 = maybe_1.__eq__(maybe_0)
    bool_2 = var_1.__eq__(var_1)
    var_4 = var_3.to_validation()
    var_5 = maybe_0.to_either()
    var_4.get_or_else(var_5)


def test_case_18():
    none_type_0 = None
    none_type_1 = None
    maybe_0 = module_0.Maybe(none_type_1, none_type_1)
    bool_0 = maybe_0.__eq__(maybe_0)
    maybe_0.map(none_type_0)
