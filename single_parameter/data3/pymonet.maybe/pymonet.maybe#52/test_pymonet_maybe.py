# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.maybe as module_0
import typing as module_1


def test_case_0():
    bool_0 = True
    generic_0 = module_0.Maybe(bool_0, bool_0)


def test_case_1():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)


def test_case_2():
    none_type_0 = None
    bool_0 = False
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    bool_1 = maybe_0.__eq__(maybe_0)


def test_case_3():
    generic_0 = module_1.Generic()
    dict_0 = {generic_0: generic_0, generic_0: generic_0, generic_0: generic_0}
    bool_0 = False
    maybe_0 = module_0.Maybe(dict_0, bool_0)
    var_0 = maybe_0.to_validation()
    var_1 = var_0.to_try()
    bool_1 = maybe_0.__eq__(var_1)
    var_1.to_either()


def test_case_4():
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.ap(none_type_0)
    var_1 = maybe_0.map(none_type_0)
    var_2 = maybe_0.ap(bool_0)
    bool_1 = var_2.__eq__(none_type_0)
    var_3 = var_2.to_box()
    var_3.bind(var_3)


def test_case_5():
    str_0 = "8Y6c3t"
    none_type_0 = None
    maybe_0 = module_0.Maybe(str_0, none_type_0)
    var_0 = maybe_0.get_or_else(none_type_0)
    maybe_0.ap(maybe_0)


def test_case_6():
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, none_type_0)
    var_0 = maybe_0.to_try()
    maybe_1 = module_0.Maybe(none_type_0, bool_0)
    var_1 = maybe_1.filter(maybe_1)
    var_2 = maybe_1.bind(none_type_0)
    maybe_0.filter(var_0)


def test_case_7():
    bytes_0 = b'\x03,"'
    str_0 = ""
    maybe_0 = module_0.Maybe(str_0, str_0)
    maybe_0.bind(bytes_0)


def test_case_8():
    generic_0 = module_1.Generic()
    dict_0 = {generic_0: generic_0, generic_0: generic_0, generic_0: generic_0}
    bool_0 = False
    maybe_0 = module_0.Maybe(dict_0, bool_0)
    var_0 = maybe_0.to_validation()
    maybe_1 = module_0.Maybe(maybe_0, generic_0)
    var_1 = maybe_1.ap(dict_0)
    var_2 = maybe_0.ap(maybe_1)
    maybe_2 = module_0.Maybe(maybe_0, var_2)
    var_3 = maybe_2.filter(var_0)
    var_0.ap(var_1)


def test_case_9():
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, none_type_0)
    var_0 = maybe_0.to_try()
    maybe_1 = module_0.Maybe(none_type_0, bool_0)
    var_1 = maybe_1.filter(maybe_1)
    maybe_0.filter(var_0)


def test_case_10():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.to_box()
    var_1 = maybe_0.get_or_else(maybe_0)
    var_2 = var_0.to_validation()
    bool_1 = False
    maybe_1 = module_0.Maybe(bool_0, bool_1)
    maybe_2 = module_0.Maybe(bool_0, var_2)
    maybe_3 = module_0.Maybe(var_0, maybe_2)
    var_3 = maybe_2.to_either()
    var_4 = maybe_3.to_either()
    var_5 = maybe_2.filter(var_0)
    var_6 = var_3.map(bool_0)
    maybe_4 = module_0.Maybe(maybe_3, var_4)
    var_7 = var_3.to_lazy()
    var_8 = maybe_4.to_either()
    var_2.map(var_8)


def test_case_11():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.get_or_else(none_type_0)
    var_0.filter(none_type_0)


def test_case_12():
    list_0 = []
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.to_either()
    var_1 = maybe_0.filter(bool_0)
    var_0.get_or_else(list_0)


def test_case_13():
    none_type_0 = None
    tuple_0 = ()
    dict_0 = {tuple_0: tuple_0, tuple_0: tuple_0, tuple_0: tuple_0}
    maybe_0 = module_0.Maybe(dict_0, tuple_0)
    var_0 = maybe_0.to_either()
    var_0.filter(none_type_0)


def test_case_14():
    list_0 = []
    bool_0 = True
    maybe_0 = module_0.Maybe(list_0, bool_0)
    var_0 = maybe_0.to_box()


def test_case_15():
    bool_0 = False
    none_type_0 = None
    maybe_0 = module_0.Maybe(bool_0, none_type_0)
    var_0 = maybe_0.to_box()
    var_1 = var_0.to_validation()
    var_1.to_validation()


def test_case_16():
    bool_0 = True
    dict_0 = {bool_0: bool_0}
    maybe_0 = module_0.Maybe(dict_0, bool_0)
    var_0 = maybe_0.to_lazy()
    maybe_1 = module_0.Maybe(dict_0, bool_0)
    var_1 = var_0.bind(var_0)
    var_2 = maybe_1.to_box()
    maybe_2 = module_0.Maybe(maybe_1, maybe_1)
    var_3 = maybe_1.bind(var_0)


def test_case_17():
    none_type_0 = None
    bool_0 = False
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    maybe_1 = module_0.Maybe(maybe_0, bool_0)
    var_0 = maybe_1.to_lazy()
    var_1 = var_0.to_validation()


def test_case_18():
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.filter(maybe_0)
    var_1 = var_0.to_try()
    none_type_1 = None
    bool_1 = False
    maybe_1 = module_0.Maybe(none_type_1, bool_1)
    maybe_1.filter(bool_0)


def test_case_19():
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    bool_1 = maybe_0.__eq__(maybe_0)
    var_0 = maybe_0.filter(maybe_0)
    var_1 = maybe_0.bind(none_type_0)
    var_2 = var_0.bind(var_0)
    var_3 = var_1.to_validation()
    var_4 = var_0.ap(var_3)
    var_5 = var_1.to_lazy()
    var_6 = var_5.to_try()
    var_7 = var_0.to_either()
    none_type_1 = None
    var_8 = var_0.filter(var_5)
    maybe_1 = module_0.Maybe(none_type_1, var_6)
    var_9 = maybe_1.to_either()
    var_10 = maybe_0.to_validation()
    var_11 = var_10.to_lazy()
    bool_2 = var_9.__eq__(var_11)


def test_case_20():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.to_validation()
    tuple_0 = ()
    maybe_1 = module_0.Maybe(tuple_0, tuple_0)
    var_1 = maybe_1.to_try()
    var_2 = var_1.get_or_else(var_0)
    var_2.to_either()


def test_case_21():
    str_0 = "m6@-<\\L"
    bool_0 = True
    maybe_0 = module_0.Maybe(str_0, bool_0)
    var_0 = maybe_0.to_lazy()
    var_1 = var_0.to_either()
    var_2 = var_1.to_box()
    none_type_0 = None
    maybe_1 = module_0.Maybe(str_0, none_type_0)
    var_3 = maybe_1.to_validation()


def test_case_22():
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, none_type_0)
    maybe_1 = module_0.Maybe(none_type_0, bool_0)
    bool_1 = maybe_1.__eq__(maybe_1)
    var_0 = maybe_1.filter(maybe_1)
    var_1 = maybe_1.bind(none_type_0)
    maybe_0.filter(maybe_1)


def test_case_23():
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.to_try()
    maybe_1 = module_0.Maybe(none_type_0, bool_0)
    bool_1 = maybe_0.__eq__(maybe_0)
    var_1 = maybe_0.filter(maybe_0)
    var_2 = maybe_1.bind(none_type_0)
    var_3 = var_1.bind(var_0)
    var_4 = maybe_1.to_box()
    var_5 = var_3.ap(var_0)
    var_6 = var_4.to_try()
    var_7 = var_1.to_lazy()
    bool_2 = False
    maybe_2 = module_0.Maybe(var_0, bool_2)
    bool_3 = maybe_2.__eq__(maybe_1)
    var_8 = var_2.to_validation()
    var_8.get_or_else(var_4)
