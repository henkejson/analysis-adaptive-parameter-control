# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.maybe as module_0


def test_case_0():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)


def test_case_1():
    bool_0 = False
    maybe_0 = module_0.Maybe(bool_0, bool_0)


def test_case_2():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    bool_0 = maybe_0.__eq__(maybe_0)


def test_case_3():
    bool_0 = True
    none_type_0 = None
    bool_1 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_1)
    bool_2 = maybe_0.__eq__(bool_0)
    maybe_1 = module_0.Maybe(bool_0, bool_0)
    bool_3 = maybe_1.__eq__(bool_2)
    var_0 = maybe_1.to_validation()
    var_1 = maybe_1.to_lazy()
    var_2 = maybe_1.to_box()
    var_3 = var_0.to_try()
    var_4 = maybe_1.bind(var_0)
    var_5 = maybe_1.map(bool_0)
    var_6 = var_4.to_try()
    var_7 = maybe_1.to_box()
    var_3.ap(var_3)


def test_case_4():
    int_0 = 1405
    none_type_0 = None
    maybe_0 = module_0.Maybe(int_0, int_0)
    var_0 = maybe_0.filter(none_type_0)
    none_type_1 = None
    var_1 = var_0.map(var_0)
    maybe_1 = module_0.Maybe(none_type_1, none_type_1)
    maybe_1.filter(var_1)


def test_case_5():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    maybe_0.ap(maybe_0)


def test_case_6():
    bytes_0 = b"\xc5\x8eb1\x19\x17oo\x86\xf3\x03\x08\xf9E\x18D\xf2\x82i\xff"
    none_type_0 = None
    maybe_0 = module_0.Maybe(bytes_0, bytes_0)
    var_0 = maybe_0.get_or_else(none_type_0)
    dict_0 = {}
    bool_0 = True
    maybe_1 = module_0.Maybe(dict_0, bool_0)
    var_1 = maybe_1.to_try()
    var_2 = maybe_1.bind(maybe_1)
    var_3 = maybe_1.filter(var_1)
    var_4 = var_3.bind(var_1)
    var_5 = maybe_0.map(var_0)
    maybe_2 = module_0.Maybe(dict_0, dict_0)
    var_6 = maybe_2.get_or_else(dict_0)
    var_7 = maybe_1.map(var_6)
    maybe_3 = module_0.Maybe(maybe_1, bytes_0)
    var_6.filter(var_0)


def test_case_7():
    bool_0 = True
    bytes_0 = b"\xa5\xb3\x01\x977\xe83\x16\xb5\xef\xcd\xab\xc7\xae"
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    bool_1 = maybe_0.__eq__(bytes_0)
    bool_2 = False
    maybe_1 = module_0.Maybe(bool_2, bool_2)
    var_0 = maybe_1.to_try()
    var_1 = maybe_0.to_validation()
    var_2 = maybe_1.to_box()
    maybe_1.bind(maybe_1)


def test_case_8():
    int_0 = 1405
    none_type_0 = None
    maybe_0 = module_0.Maybe(int_0, int_0)
    var_0 = maybe_0.filter(none_type_0)
    var_1 = var_0.ap(int_0)
    none_type_1 = None
    var_2 = var_1.map(var_1)
    maybe_1 = module_0.Maybe(none_type_1, none_type_1)
    maybe_1.filter(var_1)


def test_case_9():
    float_0 = -2860.33
    bool_0 = True
    maybe_0 = module_0.Maybe(float_0, bool_0)
    var_0 = maybe_0.get_or_else(float_0)
    maybe_1 = module_0.Maybe(var_0, bool_0)
    var_1 = maybe_1.to_either()
    var_2 = var_1.ap(float_0)
    var_2.to_either()


def test_case_10():
    str_0 = "Left[T]"
    bool_0 = True
    maybe_0 = module_0.Maybe(str_0, bool_0)
    var_0 = maybe_0.to_either()
    var_0.get_or_else(str_0)


def test_case_11():
    bool_0 = False
    dict_0 = {bool_0: bool_0, bool_0: bool_0, bool_0: bool_0, bool_0: bool_0}
    maybe_0 = module_0.Maybe(dict_0, bool_0)
    float_0 = 1513.072
    list_0 = [float_0, float_0, float_0, float_0]
    bool_1 = False
    maybe_1 = module_0.Maybe(list_0, bool_1)
    var_0 = maybe_1.to_box()
    var_1 = maybe_1.get_or_else(maybe_0)
    var_2 = maybe_1.to_box()
    var_3 = maybe_1.to_either()
    dict_1 = {}
    maybe_1.ap(dict_1)


def test_case_12():
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    bool_1 = False
    maybe_1 = module_0.Maybe(none_type_0, bool_1)
    var_0 = maybe_0.to_box()
    maybe_1.ap(var_0)


def test_case_13():
    bytes_0 = b"E\xe1\\&g\xee\x18\xef\x0f~\xff"
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, bytes_0)
    var_0 = maybe_0.to_either()
    var_1 = maybe_0.filter(var_0)
    var_2 = maybe_0.to_lazy()
    set_0 = set()
    tuple_0 = (bytes_0, set_0, bytes_0)
    var_3 = var_0.map(var_0)
    maybe_1 = module_0.Maybe(tuple_0, set_0)
    var_4 = maybe_1.to_lazy()
    var_5 = maybe_1.to_box()


def test_case_14():
    float_0 = 775.0
    bool_0 = True
    maybe_0 = module_0.Maybe(float_0, bool_0)
    var_0 = maybe_0.to_lazy()
    maybe_1 = module_0.Maybe(var_0, maybe_0)


def test_case_15():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    bool_0 = maybe_0.__eq__(maybe_0)
    var_0 = maybe_0.to_lazy()


def test_case_16():
    str_0 = ".k`Z#[ K"
    bytes_0 = b"\x8d_\x0b\x0fV\xcd\x1c\x8f\x82\xa0\\w"
    maybe_0 = module_0.Maybe(bytes_0, bytes_0)
    var_0 = maybe_0.to_try()
    var_1 = var_0.bind(str_0)
    var_1.to_lazy()


def test_case_17():
    none_type_0 = None
    bool_0 = False
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.to_try()
    var_0.bind(none_type_0)


def test_case_18():
    int_0 = 517
    str_0 = "upQ"
    maybe_0 = module_0.Maybe(str_0, str_0)
    var_0 = maybe_0.to_validation()
    var_1 = var_0.to_lazy()
    var_2 = var_1.to_box()
    var_3 = var_2.to_lazy()
    var_3.get_or_else(int_0)


def test_case_19():
    bool_0 = False
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.to_validation()
    var_1 = maybe_0.to_lazy()
    var_2 = var_1.to_either()
    maybe_0.ap(maybe_0)


def test_case_20():
    int_0 = -3111
    int_0.to_either()


def test_case_21():
    none_type_0 = None
    bool_0 = True
    bool_1 = False
    maybe_0 = module_0.Maybe(none_type_0, bool_1)
    var_0 = maybe_0.to_validation()
    var_1 = var_0.to_box()
    maybe_1 = module_0.Maybe(none_type_0, bool_0)
    var_2 = maybe_0.to_lazy()
    var_3 = maybe_1.to_lazy()
    var_4 = var_3.to_either()
    var_3.to_lazy()


def test_case_22():
    bytes_0 = b"\xf4\xe3%=\x1f\x7f\xfb\x11Z'L;\x1b\x8c\xd8"
    list_0 = [bytes_0, bytes_0, bytes_0]
    bool_0 = False
    maybe_0 = module_0.Maybe(bytes_0, bool_0)
    var_0 = maybe_0.to_lazy()
    var_1 = var_0.to_validation()
    bool_1 = False
    maybe_1 = module_0.Maybe(list_0, bool_1)


def test_case_23():
    str_0 = "S<"
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0}
    bool_0 = True
    maybe_0 = module_0.Maybe(dict_0, bool_0)
    maybe_1 = module_0.Maybe(bool_0, bool_0)
    bool_1 = maybe_1.__eq__(maybe_0)
    var_0 = maybe_1.map(maybe_0)
    var_1 = maybe_0.to_either()
    float_0 = -258.6
    var_2 = var_0.to_validation()
    var_3 = var_0.to_validation()
    bool_2 = False
    maybe_2 = module_0.Maybe(var_3, bool_2)
    var_4 = var_0.filter(var_1)
    var_5 = maybe_2.to_validation()
    maybe_2.filter(float_0)


def test_case_24():
    str_0 = "S<"
    dict_0 = {
        str_0: str_0,
        str_0: str_0,
        str_0: str_0,
        str_0: str_0,
        str_0: str_0,
        str_0: str_0,
    }
    bool_0 = True
    maybe_0 = module_0.Maybe(dict_0, bool_0)
    bool_1 = False
    maybe_1 = module_0.Maybe(bool_1, bool_1)
    bool_2 = maybe_1.__eq__(maybe_0)
    var_0 = maybe_1.to_validation()
    maybe_1.map(var_0)
