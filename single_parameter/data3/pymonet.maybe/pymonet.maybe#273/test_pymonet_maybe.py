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
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    list_0 = [bool_0, bool_0, maybe_0]
    var_0 = maybe_0.filter(maybe_0)
    bool_1 = maybe_0.__eq__(list_0)
    maybe_1 = module_0.Maybe(var_0, bool_1)
    maybe_1.filter(bool_0)


def test_case_3():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    list_0 = [bool_0, bool_0, maybe_0]
    var_0 = maybe_0.filter(maybe_0)
    bool_1 = maybe_0.__eq__(list_0)
    var_1 = maybe_0.ap(bool_0)
    var_2 = var_1.map(bool_0)
    maybe_1 = module_0.Maybe(maybe_0, var_1)
    maybe_2 = module_0.Maybe(var_0, bool_1)
    maybe_2.filter(bool_0)


def test_case_4():
    bool_0 = False
    none_type_0 = None
    maybe_0 = module_0.Maybe(bool_0, none_type_0)
    var_0 = maybe_0.get_or_else(none_type_0)
    maybe_0.map(var_0)


def test_case_5():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    list_0 = [bool_0, bool_0, maybe_0]
    var_0 = maybe_0.filter(maybe_0)
    bool_1 = maybe_0.__eq__(list_0)
    var_1 = maybe_0.ap(bool_0)
    none_type_0 = None
    var_2 = maybe_0.bind(none_type_0)
    var_3 = var_1.map(bool_0)
    var_4 = var_3.to_box()
    maybe_1 = module_0.Maybe(maybe_0, var_1)
    maybe_2 = module_0.Maybe(var_0, bool_1)
    maybe_2.filter(var_4)


def test_case_6():
    bool_0 = False
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    maybe_0.bind(bool_0)


def test_case_7():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    list_0 = [bool_0, bool_0, maybe_0]
    var_0 = maybe_0.filter(maybe_0)
    bool_1 = maybe_0.__eq__(list_0)
    var_1 = maybe_0.ap(bool_0)
    maybe_1 = module_0.Maybe(var_0, bool_1)
    maybe_1.filter(bool_0)


def test_case_8():
    bytes_0 = b"\x00j\xe74\xc3\x851\xd2\xb0J\\\\rf<!7\xb4\xf2"
    bool_0 = False
    maybe_0 = module_0.Maybe(bytes_0, bool_0)
    maybe_0.ap(maybe_0)


def test_case_9():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.to_lazy()
    float_0 = -483.11
    maybe_1 = module_0.Maybe(float_0, float_0)
    var_1 = maybe_1.get_or_else(var_0)
    var_1.to_lazy()


def test_case_10():
    bytes_0 = b"r\x07!&["
    bool_0 = True
    tuple_0 = (bool_0, bool_0)
    maybe_0 = module_0.Maybe(tuple_0, bool_0)
    var_0 = maybe_0.filter(bytes_0)
    var_1 = var_0.to_either()
    bool_1 = True
    bytes_1 = b""
    maybe_1 = module_0.Maybe(bool_1, bool_1)
    maybe_2 = module_0.Maybe(tuple_0, bytes_1)
    maybe_2.map(bytes_0)


def test_case_11():
    dict_0 = {}
    none_type_0 = None
    bool_0 = False
    maybe_0 = module_0.Maybe(dict_0, bool_0)
    bool_1 = maybe_0.__eq__(maybe_0)
    var_0 = maybe_0.to_either()
    var_1 = var_0.to_box()
    bool_2 = False
    maybe_1 = module_0.Maybe(none_type_0, bool_2)
    maybe_2 = module_0.Maybe(dict_0, dict_0)
    var_2 = maybe_2.to_lazy()
    var_3 = var_2.to_either()
    var_3.map(var_1)


def test_case_12():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    list_0 = [bool_0, bool_0, maybe_0]
    var_0 = maybe_0.filter(maybe_0)
    bool_1 = maybe_0.__eq__(list_0)
    var_1 = maybe_0.ap(bool_0)
    var_2 = var_1.map(bool_0)
    var_3 = var_2.to_box()
    maybe_1 = module_0.Maybe(maybe_0, var_1)
    maybe_2 = module_0.Maybe(var_0, bool_1)
    maybe_2.filter(var_3)


def test_case_13():
    bytes_0 = b"\xb6Q\xfcqw\x14|\xd5\xb7=R\x14"
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, bytes_0)
    var_0 = maybe_0.to_box()
    str_0 = "]:d'?-"
    maybe_1 = module_0.Maybe(bytes_0, bytes_0)
    none_type_1 = None
    maybe_2 = module_0.Maybe(str_0, none_type_1)
    var_1 = maybe_2.to_box()
    var_2 = maybe_1.to_box()
    var_1.to_box()


def test_case_14():
    bool_0 = False
    none_type_0 = None
    bool_1 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_1)
    list_0 = [bool_0, bool_0, bool_0]
    var_0 = maybe_0.ap(none_type_0)
    bool_2 = maybe_0.__eq__(var_0)
    var_1 = var_0.to_lazy()
    list_0.to_validation()


def test_case_15():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.to_lazy()
    var_1 = var_0.to_either()


def test_case_16():
    float_0 = 916.015
    dict_0 = {float_0: float_0, float_0: float_0}
    str_0 = "<'v>5R7\x0bwj Ul~"
    maybe_0 = module_0.Maybe(str_0, str_0)
    var_0 = maybe_0.to_try()
    var_1 = var_0.get_or_else(dict_0)
    var_1.to_try()


def test_case_17():
    bytes_0 = b"\xd1,\xbf\x02\xaf\x8f\xc3\xfe\xe3\xf8#T"
    none_type_0 = None
    maybe_0 = module_0.Maybe(bytes_0, none_type_0)
    var_0 = maybe_0.to_try()
    var_0.to_either()


def test_case_18():
    complex_0 = 3544.37 + 1529.19729j
    maybe_0 = module_0.Maybe(complex_0, complex_0)
    var_0 = maybe_0.to_lazy()
    var_1 = var_0.to_validation()
    var_2 = var_1.to_either()
    maybe_1 = module_0.Maybe(complex_0, complex_0)
    bool_0 = maybe_0.to_either()
    var_3 = maybe_1.to_lazy()
    var_4 = maybe_1.map(var_3)
    var_5 = maybe_1.to_either()
    var_6 = maybe_1.filter(var_5)
    var_7 = maybe_1.bind(complex_0)
    var_8 = var_4.to_validation()
    var_9 = var_5.to_try()
    var_10 = maybe_1.to_box()
    var_11 = var_10.to_lazy()
    var_12 = var_3.to_validation()


def test_case_19():
    none_type_0 = None
    str_0 = "Vc?Y4pJ"
    bool_0 = False
    maybe_0 = module_0.Maybe(str_0, bool_0)
    var_0 = maybe_0.to_validation()
    var_1 = var_0.to_box()
    var_1.get_or_else(none_type_0)


def test_case_20():
    float_0 = 4376.223
    maybe_0 = module_0.Maybe(float_0, float_0)
    var_0 = maybe_0.ap(float_0)
    var_1 = var_0.filter(var_0)
    var_2 = var_1.map(float_0)
    bool_0 = var_0.__eq__(float_0)
    bytes_0 = b"\xebdl\x13&#\xcd9\xfe_]\x87"
    var_3 = maybe_0.to_box()
    bool_1 = maybe_0.__eq__(var_0)
    bool_2 = var_0.__eq__(bool_1)
    maybe_1 = module_0.Maybe(bytes_0, bytes_0)
    var_4 = var_0.filter(var_0)
    var_5 = var_4.ap(float_0)
    var_6 = maybe_1.to_box()
    bool_3 = maybe_0.__eq__(var_6)
    var_6.ap(float_0)
