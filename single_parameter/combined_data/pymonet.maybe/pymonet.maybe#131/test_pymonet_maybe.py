# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.maybe as module_0
import typing as module_1


def test_case_0():
    str_0 = "All[value={}]"
    maybe_0 = module_0.Maybe(str_0, str_0)


def test_case_1():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)


def test_case_2():
    str_0 = "\n        Applies the function inside the Maybe[A] structure to another applicative type for notempty Maybe.\n        For empty returns copy of itself\n\n        :param applicative: applicative contains function\n        :type applicative: Maybe[B]\n        :returns: new Maybe with result of contains function\n        :rtype: Maybe[A(B) | None]\n        "
    str_1 = "4"
    maybe_0 = module_0.Maybe(str_1, str_0)
    var_0 = maybe_0.filter(str_0)
    maybe_1 = module_0.Maybe(str_1, str_1)
    bytes_0 = b"h"
    int_0 = 3371
    maybe_2 = module_0.Maybe(int_0, int_0)
    var_1 = maybe_2.get_or_else(bytes_0)
    bool_0 = maybe_0.__eq__(var_0)


def test_case_3():
    bool_0 = False
    bool_1 = True
    maybe_0 = module_0.Maybe(bool_0, bool_1)
    var_0 = maybe_0.to_validation()
    bool_2 = maybe_0.__eq__(maybe_0)
    var_1 = maybe_0.map(var_0)
    var_2 = maybe_0.to_try()
    var_3 = maybe_0.to_lazy()
    var_4 = maybe_0.ap(var_3)
    none_type_0 = None
    maybe_1 = module_0.Maybe(bool_2, bool_2)
    var_5 = maybe_1.map(var_3)
    list_0 = [var_4, var_5, maybe_1]
    bool_3 = var_1.__eq__(list_0)
    var_6 = var_4.to_try()
    var_7 = var_4.to_box()
    var_8 = var_4.ap(maybe_1)
    var_9 = maybe_0.to_lazy()
    bool_4 = maybe_0.__eq__(var_7)
    var_10 = var_6.filter(var_0)
    bool_5 = var_10.__eq__(none_type_0)
    var_11 = maybe_1.ap(var_0)
    bool_6 = none_type_0.__eq__(none_type_0)


def test_case_4():
    none_type_0 = None
    bytes_0 = b")C2[\x99\xbbS"
    bool_0 = True
    maybe_0 = module_0.Maybe(bytes_0, bool_0)
    var_0 = maybe_0.map(none_type_0)
    var_1 = var_0.to_validation()
    var_2 = var_1.to_either()


def test_case_5():
    bool_0 = False
    int_0 = -1495
    none_type_0 = None
    maybe_0 = module_0.Maybe(int_0, none_type_0)
    maybe_0.map(bool_0)


def test_case_6():
    generic_0 = module_1.Generic()
    maybe_0 = module_0.Maybe(generic_0, generic_0)
    var_0 = maybe_0.ap(maybe_0)
    var_1 = maybe_0.bind(generic_0)
    var_2 = maybe_0.to_box()
    var_3 = maybe_0.to_try()
    var_4 = maybe_0.map(var_3)
    none_type_0 = None
    maybe_1 = module_0.Maybe(var_1, none_type_0)
    var_5 = var_4.filter(var_4)
    bool_0 = False
    maybe_2 = module_0.Maybe(var_2, bool_0)
    var_3.to_lazy()


def test_case_7():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.to_validation()
    maybe_0.bind(var_0)


def test_case_8():
    tuple_0 = ()
    bool_0 = True
    maybe_0 = module_0.Maybe(tuple_0, bool_0)
    var_0 = maybe_0.ap(bool_0)
    var_1 = var_0.to_either()
    var_2 = var_0.get_or_else(tuple_0)
    var_3 = maybe_0.get_or_else(maybe_0)
    var_4 = var_3.ap(var_0)
    var_5 = var_3.map(var_0)
    var_6 = var_0.filter(var_4)
    var_7 = var_0.filter(var_1)


def test_case_9():
    bool_0 = False
    none_type_0 = None
    maybe_0 = module_0.Maybe(bool_0, none_type_0)
    bool_1 = maybe_0.__eq__(maybe_0)
    bool_2 = False
    var_0 = maybe_0.get_or_else(none_type_0)
    none_type_1 = None
    maybe_1 = module_0.Maybe(bool_2, none_type_1)
    maybe_1.ap(maybe_0)


def test_case_10():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.to_validation()
    maybe_0.filter(var_0)


def test_case_11():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.get_or_else(none_type_0)
    maybe_0.bind(var_0)


def test_case_12():
    bool_0 = False
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.to_either()


def test_case_13():
    tuple_0 = ()
    bool_0 = True
    maybe_0 = module_0.Maybe(tuple_0, bool_0)
    var_0 = maybe_0.ap(bool_0)
    var_1 = var_0.to_either()
    var_2 = var_0.get_or_else(tuple_0)
    var_3 = maybe_0.get_or_else(maybe_0)
    var_4 = var_3.ap(var_0)
    var_5 = var_3.map(var_0)
    var_6 = var_0.filter(var_4)
    var_7 = var_3.to_try()
    var_8 = maybe_0.to_validation()
    var_9 = var_1.to_try()
    var_10 = var_7.bind(var_4)
    var_11 = maybe_0.filter(var_9)
    var_12 = var_11.to_box()
    var_2.to_box()


def test_case_14():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.to_lazy()
    bytes_0 = b"\x9ab\xdcw\x801)whfS\x1f\x86Tg\xc3hn"
    bool_1 = True
    maybe_1 = module_0.Maybe(bool_1, bool_1)
    var_1 = maybe_1.to_either()
    var_1.get_or_else(bytes_0)


def test_case_15():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.to_try()
    maybe_1 = module_0.Maybe(none_type_0, none_type_0)
    var_1 = maybe_1.to_lazy()
    var_2 = maybe_1.to_either()
    bool_0 = var_1.__eq__(var_1)
    var_3 = var_2.to_lazy()
    var_4 = var_3.to_validation()
    var_5 = var_4.to_box()
    var_5.map(none_type_0)


def test_case_16():
    str_0 = "All[value={}]"
    maybe_0 = module_0.Maybe(str_0, str_0)
    var_0 = maybe_0.to_try()
    var_0.to_validation()


def test_case_17():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.to_validation()


def test_case_18():
    bool_0 = False
    bool_1 = True
    maybe_0 = module_0.Maybe(bool_0, bool_1)
    bool_2 = maybe_0.__eq__(maybe_0)
    var_0 = maybe_0.to_try()
    var_1 = maybe_0.to_lazy()
    var_2 = maybe_0.ap(var_1)
    none_type_0 = None
    var_3 = var_1.to_either()
    var_4 = maybe_0.filter(bool_1)
    var_5 = var_4.map(var_1)
    list_0 = [var_2, var_5, var_4]
    bool_3 = var_5.__eq__(list_0)
    var_6 = var_5.to_validation()
    maybe_1 = module_0.Maybe(none_type_0, none_type_0)
    var_7 = var_4.map(bool_0)
    maybe_2 = module_0.Maybe(none_type_0, none_type_0)
    var_8 = maybe_1.to_either()
    var_9 = maybe_1.to_lazy()
    var_10 = var_4.ap(var_3)
    maybe_1.filter(maybe_1)


def test_case_19():
    complex_0 = 952.733886 - 1832j
    bool_0 = True
    maybe_0 = module_0.Maybe(complex_0, bool_0)
    var_0 = maybe_0.filter(maybe_0)
    none_type_0 = None
    maybe_1 = module_0.Maybe(none_type_0, none_type_0)
    var_1 = maybe_1.to_validation()
    maybe_2 = module_0.Maybe(bool_0, complex_0)
    var_2 = maybe_1.to_box()
    maybe_1.filter(var_1)


def test_case_20():
    bool_0 = False
    bool_1 = False
    maybe_0 = module_0.Maybe(bool_0, bool_1)
    var_0 = maybe_0.to_validation()
    var_1 = module_0.Maybe(bool_1, bool_0)
    bool_2 = maybe_0.to_either()
    var_2 = var_1.to_either()
    var_3 = var_2.to_validation()
    var_4 = maybe_0.to_try()
    var_5 = maybe_0.to_lazy()
    var_6 = maybe_0.ap(var_5)
    var_7 = var_3.to_box()
    maybe_1 = module_0.Maybe(bool_2, bool_2)
    var_8 = maybe_1.map(var_5)
    list_0 = [var_6, var_8, maybe_1]
    bool_3 = var_3.__eq__(list_0)
    var_9 = var_6.to_try()
    var_4.to_box()
