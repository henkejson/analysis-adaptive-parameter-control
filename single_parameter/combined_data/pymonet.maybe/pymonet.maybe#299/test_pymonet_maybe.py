# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.maybe as module_0
import builtins as module_1


def test_case_0():
    int_0 = -1054
    maybe_0 = module_0.Maybe(int_0, int_0)


def test_case_1():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)


def test_case_2():
    none_type_0 = None
    bool_0 = False
    list_0 = [bool_0]
    bytes_0 = b"\x1aqp>\x9f'-\xa7%\xf7D\x07"
    bool_1 = True
    maybe_0 = module_0.Maybe(bytes_0, bool_1)
    var_0 = maybe_0.filter(list_0)
    var_1 = var_0.map(none_type_0)
    var_2 = var_1.to_try()
    var_3 = var_1.bind(maybe_0)
    var_4 = var_1.to_validation()
    str_0 = "G;5fk@9 /$<"
    bool_2 = False
    maybe_1 = module_0.Maybe(bool_1, var_3)
    maybe_2 = module_0.Maybe(str_0, bool_2)
    bool_3 = var_3.__eq__(maybe_0)
    var_5 = var_1.get_or_else(maybe_1)
    var_6 = maybe_2.to_lazy()
    var_7 = maybe_0.to_lazy()
    var_8 = var_1.to_box()
    var_9 = var_8.to_try()


def test_case_3():
    bytes_0 = b"V0\x18\x1bc\xe9\xe4%\xe0\tI\xa0k"
    bool_0 = False
    maybe_0 = module_0.Maybe(bytes_0, bool_0)
    var_0 = maybe_0.to_lazy()
    bool_1 = maybe_0.__eq__(bool_0)
    var_1 = var_0.ap(var_0)
    var_2 = var_0.map(var_0)
    var_3 = var_0.to_validation()
    var_4 = maybe_0.get_or_else(bytes_0)
    var_0.to_lazy()


def test_case_4():
    bool_0 = False
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    maybe_1 = module_0.Maybe(maybe_0, maybe_0)
    var_0 = maybe_1.filter(maybe_0)
    var_1 = var_0.map(maybe_0)
    var_2 = var_1.to_try()
    var_3 = var_1.to_validation()
    maybe_2 = module_0.Maybe(var_3, var_1)
    object_0 = module_1.object()
    maybe_0.filter(object_0)


def test_case_5():
    dict_0 = {}
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    maybe_0.map(dict_0)


def test_case_6():
    tuple_0 = ()
    dict_0 = {tuple_0: tuple_0}
    dict_1 = {tuple_0: dict_0}
    float_0 = 1.2508
    bool_0 = True
    maybe_0 = module_0.Maybe(float_0, bool_0)
    var_0 = maybe_0.ap(dict_0)
    var_1 = var_0.to_lazy()
    var_2 = var_1.to_either()
    var_3 = maybe_0.map(var_2)
    var_4 = maybe_0.map(var_3)
    var_5 = maybe_0.bind(dict_1)
    var_6 = var_5.to_validation()


def test_case_7():
    bytes_0 = b"\xb8j\xddU\t\x98\x97Y\x9c\xbf~^/`\xeb\xd8Y"
    bool_0 = False
    maybe_0 = module_0.Maybe(bytes_0, bool_0)
    var_0 = maybe_0.to_try()
    maybe_0.bind(var_0)


def test_case_8():
    int_0 = 1294
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.ap(int_0)
    var_1 = var_0.to_try()
    var_1.to_either()


def test_case_9():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.to_lazy()
    maybe_1 = module_0.Maybe(none_type_0, none_type_0)
    maybe_1.ap(none_type_0)


def test_case_10():
    bytes_0 = b"Y\xce\xae\xe2\xac7\x0b\xd0qQ"
    float_0 = -2556.8
    maybe_0 = module_0.Maybe(float_0, float_0)
    var_0 = maybe_0.to_either()
    var_1 = var_0.to_lazy()
    var_2 = maybe_0.filter(bytes_0)
    tuple_0 = (var_2,)
    var_3 = var_2.to_try()
    maybe_1 = module_0.Maybe(tuple_0, bytes_0)
    var_4 = maybe_1.to_try()


def test_case_11():
    object_0 = module_1.object()
    float_0 = 2702.012754
    maybe_0 = module_0.Maybe(float_0, float_0)
    var_0 = maybe_0.get_or_else(object_0)
    var_0.to_try()


def test_case_12():
    complex_0 = -479 - 110.03j
    none_type_0 = None
    maybe_0 = module_0.Maybe(complex_0, complex_0)
    var_0 = maybe_0.to_either()
    bool_0 = True
    var_1 = maybe_0.ap(none_type_0)
    maybe_1 = module_0.Maybe(none_type_0, bool_0)
    var_2 = maybe_1.get_or_else(complex_0)


def test_case_13():
    bool_0 = False
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.to_either()
    maybe_1 = module_0.Maybe(var_0, var_0)
    var_1 = maybe_1.filter(var_0)
    var_2 = var_1.map(var_0)
    var_3 = var_2.to_try()
    var_4 = var_2.to_validation()
    maybe_2 = module_0.Maybe(var_4, var_2)
    object_0 = module_1.object()
    maybe_0.filter(object_0)


def test_case_14():
    none_type_0 = None
    int_0 = 610
    maybe_0 = module_0.Maybe(none_type_0, int_0)
    var_0 = maybe_0.to_box()
    var_1 = var_0.to_either()
    var_2 = maybe_0.to_validation()
    var_3 = maybe_0.get_or_else(var_0)


def test_case_15():
    bytes_0 = b'"\x81\x14D\xda\xa2\xaa\x8c\x80\xcaf\xe0'
    none_type_0 = None
    none_type_1 = None
    maybe_0 = module_0.Maybe(bytes_0, none_type_1)
    var_0 = maybe_0.to_box()
    var_1 = var_0.to_either()
    var_1.get_or_else(none_type_0)


def test_case_16():
    bool_0 = False
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    maybe_1 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_1.to_try()
    var_1 = maybe_1.to_validation()
    bool_1 = True
    var_0.ap(bool_1)


def test_case_17():
    none_type_0 = None
    bool_0 = False
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.to_either()
    bytes_0 = b"\x1aqp>\x9f'-\xa7%\xf7D\x07"
    maybe_1 = module_0.Maybe(bytes_0, var_0)
    var_1 = maybe_1.filter(var_0)
    var_2 = var_1.map(none_type_0)
    var_3 = var_2.bind(maybe_1)
    var_4 = var_2.to_validation()
    bool_1 = False
    maybe_2 = module_0.Maybe(var_4, var_3)
    maybe_3 = module_0.Maybe(var_0, bool_1)
    bool_2 = var_3.__eq__(maybe_1)
    var_5 = var_2.to_validation()
    maybe_4 = module_0.Maybe(var_0, var_0)
    bool_3 = False
    maybe_5 = module_0.Maybe(none_type_0, bool_3)
    bool_4 = maybe_1.__eq__(var_4)
    var_6 = maybe_4.get_or_else(maybe_0)
    maybe_0.filter(var_6)


def test_case_18():
    float_0 = -677.012
    bool_0 = False
    maybe_0 = module_0.Maybe(float_0, bool_0)
    bool_1 = maybe_0.__eq__(maybe_0)
    var_0 = maybe_0.to_either()
    var_1 = var_0.to_box()
    bool_2 = True
    var_2 = maybe_0.to_try()
    maybe_1 = module_0.Maybe(float_0, bool_2)
    var_3 = maybe_1.map(var_0)
    var_4 = var_3.filter(float_0)
    var_5 = var_4.get_or_else(float_0)
    var_5.to_box()
