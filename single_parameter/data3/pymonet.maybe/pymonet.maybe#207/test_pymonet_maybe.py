# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.maybe as module_0
import typing as module_1


def test_case_0():
    str_0 = "e=|V9[qk7T*y\nqx+Eb"
    maybe_0 = module_0.Maybe(str_0, str_0)


def test_case_1():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)


def test_case_2():
    int_0 = -897
    bool_0 = False
    maybe_0 = module_0.Maybe(int_0, bool_0)
    bool_1 = maybe_0.__eq__(maybe_0)


def test_case_3():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    bool_1 = maybe_0.__eq__(bool_0)
    var_0 = maybe_0.to_either()


def test_case_4():
    bool_0 = False
    bool_1 = True
    maybe_0 = module_0.Maybe(bool_1, bool_1)
    var_0 = maybe_0.to_lazy()
    var_1 = maybe_0.map(var_0)
    var_2 = var_1.to_try()
    var_3 = var_2.get_or_else(bool_1)
    var_4 = var_1.filter(bool_0)
    var_5 = var_0.map(var_3)


def test_case_5():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.to_lazy()
    maybe_0.map(var_0)


def test_case_6():
    bool_0 = True
    none_type_0 = None
    str_0 = "inf"
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.bind(str_0)
    bool_1 = maybe_0.__eq__(var_0)


def test_case_7():
    int_0 = -897
    bool_0 = True
    maybe_0 = module_0.Maybe(int_0, bool_0)
    none_type_0 = None
    bool_1 = False
    bool_2 = maybe_0.__eq__(maybe_0)
    maybe_1 = module_0.Maybe(none_type_0, bool_1)
    var_0 = maybe_1.to_validation()
    maybe_1.bind(int_0)


def test_case_8():
    none_type_0 = None
    str_0 = 'u`*>tFst1?>(c".l'
    bool_0 = True
    maybe_0 = module_0.Maybe(str_0, bool_0)
    var_0 = maybe_0.ap(none_type_0)
    var_1 = maybe_0.to_lazy()


def test_case_9():
    bool_0 = False
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    int_0 = -1486
    maybe_0.ap(int_0)


def test_case_10():
    none_type_0 = None
    dict_0 = {none_type_0: none_type_0, none_type_0: none_type_0}
    maybe_0 = module_0.Maybe(none_type_0, dict_0)
    var_0 = maybe_0.to_lazy()
    var_1 = maybe_0.to_try()
    var_2 = maybe_0.get_or_else(none_type_0)
    var_3 = maybe_0.filter(var_2)
    var_4 = var_1.map(dict_0)


def test_case_11():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    maybe_0.filter(maybe_0)


def test_case_12():
    int_0 = -897
    bool_0 = False
    maybe_0 = module_0.Maybe(int_0, bool_0)
    bool_1 = maybe_0.__eq__(maybe_0)
    var_0 = maybe_0.get_or_else(bool_1)
    var_0.to_box()


def test_case_13():
    int_0 = -897
    bool_0 = True
    maybe_0 = module_0.Maybe(int_0, bool_0)
    none_type_0 = None
    bool_1 = False
    bool_2 = maybe_0.__eq__(maybe_0)
    maybe_1 = module_0.Maybe(none_type_0, bool_1)
    var_0 = maybe_1.to_validation()
    var_1 = maybe_1.to_either()
    maybe_1.bind(int_0)


def test_case_14():
    int_0 = -167
    bool_0 = True
    maybe_0 = module_0.Maybe(int_0, bool_0)
    var_0 = maybe_0.to_box()
    maybe_1 = module_0.Maybe(int_0, int_0)


def test_case_15():
    list_0 = []
    bool_0 = False
    maybe_0 = module_0.Maybe(list_0, bool_0)
    var_0 = maybe_0.to_box()
    maybe_0.ap(list_0)


def test_case_16():
    bool_0 = False
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    bool_1 = maybe_0.__eq__(maybe_0)
    var_0 = maybe_0.to_lazy()
    bool_2 = var_0.__eq__(var_0)
    var_1 = maybe_0.to_try()
    maybe_0.map(var_1)


def test_case_17():
    float_0 = -1733.659
    maybe_0 = module_0.Maybe(float_0, float_0)
    var_0 = maybe_0.to_validation()


def test_case_18():
    none_type_0 = None
    bool_0 = False
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.to_validation()


def test_case_19():
    str_0 = 'z{3^y#"Nh1w-='
    generic_0 = module_1.Generic()
    dict_0 = {generic_0: generic_0}
    bool_0 = True
    maybe_0 = module_0.Maybe(dict_0, bool_0)
    var_0 = maybe_0.to_lazy()
    var_1 = var_0.to_validation()
    var_1.map(str_0)


def test_case_20():
    none_type_0 = None
    bool_0 = False
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    bool_1 = maybe_0.__eq__(bool_0)
    var_0 = maybe_0.to_lazy()
    var_1 = var_0.to_validation()
    var_2 = maybe_0.get_or_else(maybe_0)
    maybe_0.ap(maybe_0)


def test_case_21():
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    none_type_1 = None
    bool_1 = maybe_0.__eq__(none_type_0)
    var_0 = maybe_0.to_lazy()
    var_1 = maybe_0.ap(bool_0)
    var_2 = var_1.to_lazy()
    var_3 = maybe_0.bind(var_2)
    maybe_1 = module_0.Maybe(none_type_1, none_type_1)
    var_4 = var_1.bind(var_3)
    var_5 = var_4.filter(var_1)
    var_6 = var_1.ap(none_type_0)
    var_7 = var_5.to_lazy()
    bool_2 = var_6.__eq__(maybe_0)
    bool_3 = maybe_1.__eq__(var_6)
    var_8 = var_5.to_lazy()
    var_9 = var_5.map(var_8)
    maybe_1.filter(var_8)
