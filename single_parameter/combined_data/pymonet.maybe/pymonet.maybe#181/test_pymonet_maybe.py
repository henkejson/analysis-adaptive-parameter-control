# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.maybe as module_0
import typing as module_1


def test_case_0():
    int_0 = -1043
    maybe_0 = module_0.Maybe(int_0, int_0)


def test_case_1():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)


def test_case_2():
    dict_0 = {}
    bool_0 = False
    maybe_0 = module_0.Maybe(dict_0, bool_0)
    var_0 = maybe_0.to_either()
    bool_1 = maybe_0.__eq__(maybe_0)
    var_1 = maybe_0.to_try()
    var_1.to_try()


def test_case_3():
    str_0 = "Ie[Oblf"
    maybe_0 = module_0.Maybe(str_0, str_0)
    var_0 = maybe_0.to_validation()
    var_1 = var_0.to_lazy()
    bool_0 = maybe_0.__eq__(var_1)
    var_0.to_validation()


def test_case_4():
    bytes_0 = b"M\xc8\x87\xc9\xce\xbd5\x92\xffV\x94R\x96\x91\xea\xea"
    str_0 = "O;6\\tvoSWWdbN#NWA"
    str_1 = "[4_4`AQy5(LBq[H+4p"
    float_0 = -1588.1247
    list_0 = [float_0, float_0]
    bool_0 = True
    maybe_0 = module_0.Maybe(list_0, bool_0)
    var_0 = maybe_0.ap(str_1)
    var_1 = var_0.map(str_0)
    var_2 = var_1.get_or_else(bytes_0)


def test_case_5():
    bytes_0 = b"\x07;X\x9f*\x84\xa3\xa3I.\x06-\x83\x1b\xb2"
    bool_0 = False
    maybe_0 = module_0.Maybe(bytes_0, bool_0)
    maybe_0.map(bytes_0)


def test_case_6():
    none_type_0 = None
    str_0 = "BL9%Xw"
    str_1 = "Cnqu~Rlkqs"
    bool_0 = True
    maybe_0 = module_0.Maybe(str_1, bool_0)
    var_0 = maybe_0.bind(str_0)
    var_1 = var_0.to_either()
    var_2 = var_1.to_validation()
    var_3 = var_2.to_try()
    var_4 = var_3.get_or_else(none_type_0)


def test_case_7():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    bool_0 = maybe_0.__eq__(none_type_0)
    var_0 = maybe_0.to_validation()
    var_1 = var_0.to_try()
    maybe_0.bind(none_type_0)


def test_case_8():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    maybe_0.ap(none_type_0)


def test_case_9():
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.to_validation()
    var_1 = maybe_0.filter(none_type_0)


def test_case_10():
    none_type_0 = None
    bool_0 = False
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.to_validation()
    maybe_0.filter(none_type_0)


def test_case_11():
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    none_type_1 = None
    bool_1 = True
    maybe_1 = module_0.Maybe(none_type_1, bool_1)
    var_0 = maybe_1.get_or_else(maybe_0)


def test_case_12():
    none_type_0 = None
    bool_0 = False
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    bool_1 = False
    none_type_1 = None
    maybe_1 = module_0.Maybe(bool_1, none_type_1)
    var_0 = maybe_1.get_or_else(maybe_0)
    var_0.get_or_else(none_type_0)


def test_case_13():
    int_0 = -1127
    maybe_0 = module_0.Maybe(int_0, int_0)
    var_0 = maybe_0.to_either()
    bool_0 = True
    maybe_1 = module_0.Maybe(maybe_0, bool_0)
    var_1 = maybe_0.to_box()
    var_2 = var_0.to_lazy()
    var_3 = var_0.to_validation()
    var_4 = var_0.to_lazy()


def test_case_14():
    float_0 = 1706.291
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.to_either()
    maybe_1 = module_0.Maybe(float_0, float_0)
    var_1 = maybe_1.to_validation()
    bool_0 = var_1.__eq__(maybe_1)


def test_case_15():
    list_0 = []
    none_type_0 = None
    bool_0 = False
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.to_box()
    bool_1 = True
    maybe_1 = module_0.Maybe(var_0, bool_1)
    var_1 = maybe_1.map(none_type_0)
    var_2 = var_1.to_either()
    var_3 = var_2.ap(list_0)
    maybe_2 = module_0.Maybe(var_3, var_0)


def test_case_16():
    int_0 = -1043
    maybe_0 = module_0.Maybe(int_0, int_0)
    var_0 = maybe_0.to_lazy()


def test_case_17():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.to_lazy()
    var_0.get_or_else(var_0)


def test_case_18():
    str_0 = "g\x0c;E`"
    bool_0 = True
    maybe_0 = module_0.Maybe(str_0, bool_0)
    var_0 = maybe_0.to_try()
    var_1 = var_0.filter(str_0)
    maybe_1 = module_0.Maybe(str_0, str_0)
    var_2 = maybe_1.to_validation()
    var_3 = var_2.to_lazy()
    bool_1 = maybe_1.__eq__(var_3)
    var_2.to_validation()


def test_case_19():
    dict_0 = {}
    maybe_0 = module_0.Maybe(dict_0, dict_0)
    var_0 = maybe_0.to_try()
    maybe_0.filter(maybe_0)


def test_case_20():
    generic_0 = module_1.Generic()
    bool_0 = True
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    maybe_1 = module_0.Maybe(generic_0, bool_0)
    none_type_1 = None
    var_0 = maybe_1.bind(none_type_1)
    var_1 = maybe_1.map(generic_0)
    var_2 = var_1.ap(generic_0)
    var_3 = var_2.to_box()
    var_4 = var_0.to_lazy()
    generic_1 = module_1.Generic()
    var_5 = var_3.to_either()
    set_0 = {generic_1}
    maybe_2 = module_0.Maybe(set_0, set_0)
    bool_1 = maybe_1.__eq__(maybe_0)
    var_6 = maybe_2.ap(none_type_1)
    var_7 = var_2.filter(var_6)
    var_8 = var_7.map(var_6)
    var_9 = maybe_2.to_validation()
    list_0 = [set_0, bool_0, bool_0]
    var_10 = var_4.map(list_0)
    var_10.to_validation()


def test_case_21():
    bytes_0 = b"\xcbsL\x95\x11\x81)4\xf0\xd4I8"
    bool_0 = False
    maybe_0 = module_0.Maybe(bytes_0, bool_0)
    var_0 = maybe_0.to_lazy()
    var_1 = var_0.to_try()
    var_1.filter(bytes_0)


def test_case_22():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    none_type_0 = None
    maybe_1 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_1.to_try()
    var_1 = maybe_1.filter(var_0)
    bool_1 = maybe_1.__eq__(var_1)
    var_0.ap(none_type_0)
