# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.maybe as module_0
import typing as module_1
import builtins as module_2


def test_case_0():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)


def test_case_1():
    tuple_0 = ()
    maybe_0 = module_0.Maybe(tuple_0, tuple_0)


def test_case_2():
    none_type_0 = None
    bytes_0 = b""
    maybe_0 = module_0.Maybe(bytes_0, bytes_0)
    bool_0 = maybe_0.__eq__(maybe_0)
    bool_1 = False
    maybe_1 = module_0.Maybe(maybe_0, bool_1)
    var_0 = maybe_1.get_or_else(none_type_0)
    maybe_0.ap(none_type_0)


def test_case_3():
    str_0 = "M$[Ard54-GlaZ?[h"
    maybe_0 = module_0.Maybe(str_0, str_0)
    bool_0 = maybe_0.__eq__(str_0)
    var_0 = maybe_0.to_either()
    var_0.get_or_else(str_0)


def test_case_4():
    int_0 = -1664
    generic_0 = module_1.Generic()
    bool_0 = True
    maybe_0 = module_0.Maybe(generic_0, bool_0)
    var_0 = maybe_0.map(int_0)
    var_1 = var_0.to_box()


def test_case_5():
    bool_0 = False
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    bool_1 = maybe_0.__eq__(bool_0)
    maybe_0.map(maybe_0)


def test_case_6():
    list_0 = []
    bool_0 = True
    maybe_0 = module_0.Maybe(list_0, bool_0)
    int_0 = -1405
    var_0 = maybe_0.to_validation()
    maybe_1 = module_0.Maybe(int_0, int_0)
    var_1 = maybe_1.map(var_0)
    var_2 = maybe_1.bind(var_0)
    var_3 = maybe_1.to_lazy()
    var_4 = var_3.to_box()
    var_5 = maybe_1.to_either()


def test_case_7():
    none_type_0 = None
    int_0 = 2453
    bool_0 = False
    maybe_0 = module_0.Maybe(int_0, bool_0)
    maybe_0.bind(none_type_0)


def test_case_8():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.filter(maybe_0)
    var_1 = maybe_0.map(maybe_0)
    bool_1 = var_0.__eq__(var_0)
    var_2 = maybe_0.to_try()
    var_3 = var_1.to_box()
    var_4 = maybe_0.map(var_0)
    var_5 = maybe_0.ap(var_3)
    var_6 = var_4.get_or_else(var_3)
    var_7 = var_4.map(var_5)
    var_8 = maybe_0.to_try()


def test_case_9():
    int_0 = 3251
    bool_0 = False
    maybe_0 = module_0.Maybe(int_0, bool_0)
    maybe_0.filter(bool_0)


def test_case_10():
    tuple_0 = ()
    bool_0 = True
    none_type_0 = None
    set_0 = {tuple_0, none_type_0, bool_0}
    maybe_0 = module_0.Maybe(none_type_0, set_0)
    var_0 = maybe_0.to_either()
    maybe_1 = module_0.Maybe(tuple_0, bool_0)


def test_case_11():
    bool_0 = False
    none_type_0 = None
    maybe_0 = module_0.Maybe(bool_0, none_type_0)
    var_0 = maybe_0.to_either()
    var_1 = var_0.to_box()
    maybe_0.filter(var_1)


def test_case_12():
    tuple_0 = ()
    dict_0 = {tuple_0: tuple_0, tuple_0: tuple_0}
    bool_0 = True
    none_type_0 = None
    maybe_0 = module_0.Maybe(bool_0, none_type_0)
    var_0 = maybe_0.to_box()
    var_0.get_or_else(dict_0)


def test_case_13():
    none_type_0 = None
    bytes_0 = b"\x02\x1b\nY \x14$\xdc\xc0^\x9a[q"
    maybe_0 = module_0.Maybe(none_type_0, bytes_0)
    var_0 = maybe_0.filter(none_type_0)
    bool_0 = maybe_0.__eq__(maybe_0)
    maybe_1 = module_0.Maybe(var_0, none_type_0)
    var_1 = maybe_1.to_lazy()
    var_2 = maybe_1.get_or_else(maybe_0)
    var_3 = var_1.to_either()
    var_4 = maybe_0.map(none_type_0)
    var_5 = var_2.bind(var_3)
    maybe_1.filter(var_3)


def test_case_14():
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.to_try()


def test_case_15():
    object_0 = module_2.object()
    bool_0 = True
    maybe_0 = module_0.Maybe(object_0, bool_0)
    var_0 = maybe_0.to_try()
    none_type_0 = None
    bool_1 = False
    maybe_1 = module_0.Maybe(none_type_0, bool_1)
    var_1 = maybe_1.to_box()
    var_2 = maybe_1.to_try()
    var_2.to_box()


def test_case_16():
    float_0 = -26.7
    none_type_0 = None
    none_type_1 = None
    maybe_0 = module_0.Maybe(float_0, none_type_1)
    var_0 = maybe_0.to_validation()
    maybe_0.filter(none_type_0)


def test_case_17():
    str_0 = "zQ\\$c'o> 4H\t:$9c^"
    maybe_0 = module_0.Maybe(str_0, str_0)
    var_0 = maybe_0.filter(maybe_0)
    var_1 = maybe_0.filter(var_0)
    var_2 = maybe_0.map(var_0)
    bool_0 = maybe_0.__eq__(var_0)
    var_3 = var_2.to_try()
    var_4 = var_1.to_validation()
    var_3.to_box()
