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
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.ap(none_type_0)
    str_0 = "v"
    bool_1 = maybe_0.__eq__(str_0)
    var_1 = maybe_0.map(var_0)


def test_case_3():
    none_type_0 = None
    none_type_1 = None
    maybe_0 = module_0.Maybe(none_type_1, none_type_1)
    maybe_0.map(none_type_0)


def test_case_4():
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.to_try()
    bool_1 = True
    maybe_1 = module_0.Maybe(bool_1, bool_1)
    var_1 = maybe_0.bind(var_0)
    var_2 = maybe_1.to_validation()
    var_3 = var_2.to_either()
    var_4 = var_3.to_validation()
    var_5 = var_4.to_box()
    maybe_2 = module_0.Maybe(var_5, bool_0)
    var_5.map(var_0)


def test_case_5():
    int_0 = 5903
    maybe_0 = module_0.Maybe(int_0, int_0)
    var_0 = maybe_0.filter(maybe_0)
    var_1 = maybe_0.ap(maybe_0)


def test_case_6():
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, none_type_0)
    bool_1 = maybe_0.__eq__(bool_0)
    var_0 = maybe_0.to_validation()
    maybe_0.ap(maybe_0)


def test_case_7():
    str_0 = "`-{\\w4qEbQ^ps"
    bool_0 = False
    tuple_0 = ()
    str_1 = "d"
    tuple_1 = (bool_0, tuple_0, bool_0, str_1)
    dict_0 = {tuple_1: str_1, bool_0: str_1, tuple_0: tuple_0}
    maybe_0 = module_0.Maybe(dict_0, tuple_1)
    var_0 = maybe_0.bind(str_0)
    maybe_1 = module_0.Maybe(bool_0, bool_0)
    var_1 = maybe_1.get_or_else(bool_0)
    maybe_1.filter(var_1)


def test_case_8():
    str_0 = "`-{\\w4qEbQ^ps"
    bool_0 = False
    tuple_0 = ()
    str_1 = "d"
    tuple_1 = (bool_0, tuple_0, bool_0, str_1)
    dict_0 = {tuple_1: str_1, bool_0: str_1, tuple_0: tuple_0}
    maybe_0 = module_0.Maybe(dict_0, tuple_1)
    var_0 = maybe_0.bind(str_0)
    bool_1 = True
    maybe_1 = module_0.Maybe(bool_1, bool_1)
    var_1 = maybe_1.get_or_else(bool_1)
    var_2 = maybe_1.filter(var_1)
    var_3 = maybe_1.to_box()
    var_4 = var_3.to_validation()
    var_5 = var_4.to_try()


def test_case_9():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.to_either()


def test_case_10():
    bool_0 = False
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.to_try()
    var_1 = maybe_0.to_either()
    var_0.to_validation()


def test_case_11():
    bool_0 = False
    set_0 = {bool_0}
    list_0 = [bool_0, set_0, set_0, set_0]
    bytes_0 = b"\x84\x99h\x19E\x90\x8d\xf7\x82\xb3\xa7\xf6"
    tuple_0 = (list_0, bytes_0, set_0)
    maybe_0 = module_0.Maybe(bool_0, list_0)
    var_0 = maybe_0.ap(bool_0)
    var_1 = var_0.to_either()
    list_1 = [tuple_0, var_1, list_0, var_0]
    maybe_1 = module_0.Maybe(list_1, var_0)
    var_2 = maybe_1.to_box()


def test_case_12():
    bool_0 = True
    none_type_0 = None
    maybe_0 = module_0.Maybe(bool_0, none_type_0)
    var_0 = maybe_0.to_validation()
    var_1 = var_0.to_either()
    bytes_0 = b"\xad\x90\x8a\x12\x00\xfa"
    bool_1 = False
    maybe_1 = module_0.Maybe(bytes_0, bool_1)
    tuple_0 = ()
    var_2 = maybe_1.to_either()
    tuple_1 = (bytes_0, maybe_1, bool_1, tuple_0)
    var_3 = maybe_1.to_either()
    bool_2 = True
    maybe_2 = module_0.Maybe(tuple_1, bool_2)
    bool_3 = False
    maybe_3 = module_0.Maybe(var_2, bool_3)
    var_4 = var_2.to_box()
    var_5 = maybe_1.to_try()
    maybe_4 = module_0.Maybe(maybe_2, bool_1)
    var_6 = maybe_4.to_box()
    var_7 = maybe_1.to_either()
    var_8 = var_6.to_try()
    none_type_1 = None
    bool_4 = False
    var_9 = maybe_2.ap(var_7)
    maybe_5 = module_0.Maybe(none_type_1, bool_4)
    var_10 = maybe_2.to_either()


def test_case_13():
    int_0 = 5903
    maybe_0 = module_0.Maybe(int_0, int_0)
    var_0 = maybe_0.filter(maybe_0)
    maybe_1 = module_0.Maybe(var_0, int_0)
    bool_0 = True
    maybe_2 = module_0.Maybe(maybe_0, bool_0)
    var_1 = var_0.to_either()
    var_2 = var_1.ap(bool_0)
    var_3 = maybe_0.to_validation()
    var_4 = maybe_0.to_try()
    var_5 = maybe_0.to_lazy()


def test_case_14():
    tuple_0 = ()
    maybe_0 = module_0.Maybe(tuple_0, tuple_0)
    none_type_0 = None
    maybe_1 = module_0.Maybe(none_type_0, none_type_0)
    int_0 = -1232
    var_0 = maybe_1.to_lazy()
    bool_0 = var_0.__eq__(int_0)


def test_case_15():
    str_0 = "E^g}'"
    bool_0 = False
    bool_1 = True
    maybe_0 = module_0.Maybe(str_0, bool_1)
    var_0 = maybe_0.to_try()
    bool_2 = False
    maybe_1 = module_0.Maybe(bool_0, bool_2)
    var_1 = maybe_1.to_try()
    var_1.map(str_0)


def test_case_16():
    int_0 = 0
    set_0 = {int_0, int_0, int_0}
    none_type_0 = None
    maybe_0 = module_0.Maybe(int_0, none_type_0)
    maybe_1 = module_0.Maybe(int_0, int_0)
    var_0 = maybe_1.to_validation()
    var_0.map(set_0)


def test_case_17():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.to_lazy()
    var_1 = maybe_0.to_lazy()
    var_2 = var_0.to_try()
    var_1.to_lazy()


def test_case_18():
    dict_0 = {}
    maybe_0 = module_0.Maybe(dict_0, dict_0)
    maybe_1 = module_0.Maybe(dict_0, dict_0)
    var_0 = maybe_1.to_lazy()
    var_1 = var_0.to_either()
    var_2 = var_0.to_either()
    maybe_2 = module_0.Maybe(maybe_0, var_0)
    var_3 = maybe_2.to_try()
    bool_0 = var_3.__eq__(maybe_0)
    var_1.to_either()


def test_case_19():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    maybe_0.bind(none_type_0)


def test_case_20():
    bool_0 = False
    bool_1 = True
    maybe_0 = module_0.Maybe(bool_0, bool_1)
    var_0 = maybe_0.to_either()
    var_1 = maybe_0.bind(bool_1)
    var_2 = var_1.map(var_0)
    var_3 = var_1.map(var_0)
    var_4 = var_1.bind(var_0)
    bool_2 = maybe_0.__eq__(bool_0)
    var_5 = maybe_0.to_either()
    var_6 = var_1.filter(var_0)
    bool_3 = var_1.__eq__(var_1)
    var_7 = var_1.filter(var_5)
    bool_4 = False
    var_8 = var_1.bind(bool_4)
    var_9 = var_1.get_or_else(var_6)
    var_10 = var_4.map(var_0)
    bool_3.to_validation()


def test_case_21():
    float_0 = 982.723
    maybe_0 = module_0.Maybe(float_0, float_0)
    var_0 = maybe_0.to_either()
    var_1 = maybe_0.filter(var_0)
    var_2 = maybe_0.filter(var_0)
    bool_0 = maybe_0.__eq__(var_2)
    var_0.get_or_else(maybe_0)


def test_case_22():
    bool_0 = False
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    bool_1 = maybe_0.__eq__(maybe_0)
    maybe_0.ap(bool_0)
