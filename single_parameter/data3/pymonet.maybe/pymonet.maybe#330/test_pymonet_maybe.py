# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.maybe as module_0
import typing as module_1


def test_case_0():
    bytes_0 = b"@4\x19="
    maybe_0 = module_0.Maybe(bytes_0, bytes_0)


def test_case_1():
    bool_0 = False
    maybe_0 = module_0.Maybe(bool_0, bool_0)


def test_case_2():
    float_0 = -847.007033
    maybe_0 = module_0.Maybe(float_0, float_0)
    bool_0 = maybe_0.__eq__(float_0)
    var_0 = maybe_0.to_try()
    var_0.to_try()


def test_case_3():
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.to_lazy()
    var_1 = maybe_0.map(none_type_0)
    var_2 = maybe_0.to_either()
    var_3 = maybe_0.bind(none_type_0)
    bool_1 = var_0.__eq__(var_2)
    var_4 = var_0.map(maybe_0)


def test_case_4():
    none_type_0 = None
    bool_0 = False
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    maybe_0.map(none_type_0)


def test_case_5():
    int_0 = 1254
    bool_0 = False
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    maybe_0.bind(int_0)


def test_case_6():
    none_type_0 = None
    int_0 = -1641
    maybe_0 = module_0.Maybe(none_type_0, int_0)
    var_0 = maybe_0.to_try()
    maybe_1 = module_0.Maybe(int_0, int_0)
    var_1 = maybe_1.to_box()
    var_2 = var_1.to_validation()
    var_3 = maybe_1.ap(int_0)
    var_4 = var_3.map(var_3)
    var_5 = var_4.ap(none_type_0)
    var_6 = var_1.to_try()
    var_7 = maybe_1.map(none_type_0)


def test_case_7():
    str_0 = "CC&;}p;m_$/E0\t:?x-q"
    maybe_0 = module_0.Maybe(str_0, str_0)
    var_0 = maybe_0.to_lazy()
    bool_0 = False
    maybe_1 = module_0.Maybe(str_0, bool_0)
    var_1 = maybe_1.to_lazy()
    maybe_1.ap(str_0)


def test_case_8():
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.to_lazy()
    var_1 = maybe_0.map(none_type_0)
    var_2 = maybe_0.to_either()
    var_3 = maybe_0.bind(none_type_0)
    bool_1 = False
    maybe_1 = module_0.Maybe(bool_1, bool_1)
    bool_2 = var_0.__eq__(var_2)
    var_4 = var_0.map(maybe_0)
    var_5 = var_3.filter(var_2)


def test_case_9():
    tuple_0 = ()
    bool_0 = False
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    maybe_0.filter(tuple_0)


def test_case_10():
    bytes_0 = b"\x18\xe0\xaf;\x0fU "
    none_type_0 = None
    maybe_0 = module_0.Maybe(bytes_0, bytes_0)
    var_0 = maybe_0.get_or_else(none_type_0)
    bool_0 = False
    maybe_1 = module_0.Maybe(bytes_0, bool_0)
    var_1 = maybe_1.to_validation()
    var_2 = var_1.to_box()
    var_2.to_box()


def test_case_11():
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.to_lazy()
    var_1 = maybe_0.map(none_type_0)
    var_2 = maybe_0.to_either()
    var_3 = maybe_0.bind(none_type_0)
    bool_1 = False
    maybe_1 = module_0.Maybe(bool_1, bool_1)
    bool_2 = True
    var_4 = maybe_1.get_or_else(bool_2)
    bool_3 = var_0.__eq__(var_2)
    var_5 = var_0.map(maybe_0)
    var_6 = var_3.filter(var_2)


def test_case_12():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.to_box()
    var_1 = maybe_0.to_try()
    var_2 = var_0.to_lazy()


def test_case_13():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    bool_0 = maybe_0.__eq__(maybe_0)
    var_0 = maybe_0.to_box()
    var_1 = maybe_0.to_either()
    var_1.ap(maybe_0)


def test_case_14():
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.to_try()


def test_case_15():
    bool_0 = False
    list_0 = [bool_0]
    bool_1 = False
    maybe_0 = module_0.Maybe(list_0, bool_0)
    str_0 = "nv](Xqc.'yg4 \r"
    bool_2 = maybe_0.__eq__(str_0)
    var_0 = maybe_0.to_try()
    none_type_0 = None
    maybe_1 = module_0.Maybe(bool_1, none_type_0)
    var_1 = maybe_1.to_try()
    var_2 = var_1.get_or_else(var_1)
    var_1.to_lazy()


def test_case_16():
    bytes_0 = b"@4\x19="
    bool_0 = True
    maybe_0 = module_0.Maybe(bytes_0, bool_0)
    var_0 = maybe_0.to_try()
    var_1 = maybe_0.to_validation()
    var_2 = var_1.to_try()
    var_2.to_try()


def test_case_17():
    int_0 = -1719
    generic_0 = module_1.Generic()
    bool_0 = True
    maybe_0 = module_0.Maybe(int_0, bool_0)
    bool_1 = maybe_0.__eq__(maybe_0)
    maybe_1 = module_0.Maybe(int_0, generic_0)
    str_0 = "RSn]0T#07mz`]oArl3"
    var_0 = maybe_0.get_or_else(bool_1)
    var_1 = maybe_1.map(var_0)
    none_type_0 = None
    var_2 = maybe_1.bind(none_type_0)
    var_3 = var_2.to_validation()
    var_4 = var_1.bind(var_0)
    var_5 = maybe_0.to_lazy()
    var_6 = maybe_0.to_lazy()
    var_7 = maybe_0.filter(maybe_1)
    maybe_2 = module_0.Maybe(none_type_0, bool_0)
    maybe_3 = module_0.Maybe(str_0, bool_0)
    var_8 = var_2.to_try()
    var_9 = var_1.bind(var_1)
    var_10 = var_2.to_either()
    var_11 = var_5.to_either()
    bool_2 = var_7.__eq__(bool_1)
    var_12 = maybe_2.to_box()
    var_13 = var_4.to_lazy()


def test_case_18():
    bytes_0 = b"\x0e\xaa\xe5\xbck\x12=\xf2EF"
    dict_0 = {bytes_0: bytes_0, bytes_0: bytes_0}
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    bool_1 = maybe_0.__eq__(none_type_0)
    maybe_1 = module_0.Maybe(dict_0, none_type_0)
    var_0 = maybe_1.to_either()
    var_1 = var_0.to_try()
    bool_2 = True
    var_2 = var_0.to_try()
    maybe_2 = module_0.Maybe(dict_0, bool_2)
    var_3 = maybe_2.to_either()
    var_4 = maybe_2.to_either()
    var_5 = maybe_2.to_validation()
    var_6 = var_3.to_try()
    var_1.to_either()


def test_case_19():
    int_0 = -1719
    maybe_0 = module_0.Maybe(int_0, int_0)
    bool_0 = maybe_0.__eq__(maybe_0)
    maybe_1 = module_0.Maybe(int_0, maybe_0)
    var_0 = maybe_0.get_or_else(bool_0)
    var_1 = maybe_1.map(var_0)
    none_type_0 = None
    var_2 = maybe_1.bind(none_type_0)
    var_3 = maybe_0.to_lazy()
    var_4 = var_1.filter(bool_0)
    maybe_2 = module_0.Maybe(var_4, none_type_0)
    maybe_3 = module_0.Maybe(var_1, bool_0)
    maybe_2.filter(maybe_3)


def test_case_20():
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, none_type_0)
    var_0 = maybe_0.to_lazy()
    var_1 = var_0.map(var_0)
    var_1.to_either()


def test_case_21():
    int_0 = -1719
    generic_0 = module_1.Generic()
    bool_0 = True
    maybe_0 = module_0.Maybe(int_0, bool_0)
    bool_1 = maybe_0.__eq__(maybe_0)
    maybe_1 = module_0.Maybe(int_0, generic_0)
    var_0 = maybe_0.get_or_else(bool_1)
    var_1 = maybe_1.map(var_0)
    none_type_0 = None
    var_2 = maybe_1.bind(none_type_0)
    var_3 = var_2.to_validation()
    var_4 = var_1.bind(var_0)
    var_5 = maybe_0.to_lazy()
    var_6 = var_4.filter(bool_1)
    maybe_2 = module_0.Maybe(var_6, none_type_0)
    bool_2 = True
    maybe_3 = module_0.Maybe(bool_1, bool_2)
    var_7 = maybe_1.to_try()
    var_8 = var_6.to_try()
    var_9 = var_4.bind(var_8)
    var_10 = var_6.bind(var_5)
    var_11 = var_10.to_either()
    var_12 = var_10.to_either()
    bool_3 = maybe_2.__eq__(var_2)
    var_13 = var_2.to_box()
    var_14 = var_9.to_lazy()
