# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typing as module_0
import pymonet.maybe as module_1


def test_case_0():
    generic_0 = module_0.Generic()
    maybe_0 = module_1.Maybe(generic_0, generic_0)


def test_case_1():
    none_type_0 = None
    maybe_0 = module_1.Maybe(none_type_0, none_type_0)


def test_case_2():
    complex_0 = -143 + 852.481j
    bool_0 = True
    maybe_0 = module_1.Maybe(complex_0, bool_0)
    none_type_0 = None
    var_0 = maybe_0.filter(none_type_0)
    bool_1 = var_0.__eq__(maybe_0)
    var_1 = maybe_0.bind(complex_0)
    var_2 = var_1.to_box()
    bool_0.to_validation()


def test_case_3():
    none_type_0 = None
    none_type_1 = None
    maybe_0 = module_1.Maybe(none_type_1, none_type_1)
    var_0 = maybe_0.to_box()
    var_1 = maybe_0.to_either()
    bool_0 = maybe_0.__eq__(none_type_0)
    var_2 = var_0.to_either()


def test_case_4():
    int_0 = 4611
    bool_0 = True
    maybe_0 = module_1.Maybe(int_0, bool_0)
    var_0 = maybe_0.map(bool_0)
    var_1 = maybe_0.to_box()


def test_case_5():
    none_type_0 = None
    bool_0 = False
    maybe_0 = module_1.Maybe(none_type_0, bool_0)
    maybe_0.map(none_type_0)


def test_case_6():
    str_0 = "1_|7Nqq\t=ELd\\H9!qdc>"
    bool_0 = False
    maybe_0 = module_1.Maybe(str_0, bool_0)
    maybe_0.bind(maybe_0)


def test_case_7():
    generic_0 = module_0.Generic()
    maybe_0 = module_1.Maybe(generic_0, generic_0)
    none_type_0 = None
    var_0 = maybe_0.map(none_type_0)
    var_1 = maybe_0.filter(generic_0)
    var_2 = maybe_0.bind(generic_0)
    var_3 = var_2.ap(generic_0)
    var_4 = var_3.get_or_else(generic_0)
    var_4.to_validation()


def test_case_8():
    int_0 = -941
    none_type_0 = None
    maybe_0 = module_1.Maybe(none_type_0, none_type_0)
    maybe_0.ap(int_0)


def test_case_9():
    bool_0 = False
    maybe_0 = module_1.Maybe(bool_0, bool_0)
    var_0 = maybe_0.to_validation()
    maybe_0.filter(var_0)


def test_case_10():
    list_0 = []
    int_0 = 1612
    dict_0 = {int_0: int_0, int_0: int_0, int_0: int_0, int_0: int_0}
    bool_0 = False
    maybe_0 = module_1.Maybe(dict_0, bool_0)
    var_0 = maybe_0.to_lazy()
    var_1 = maybe_0.get_or_else(list_0)
    var_1.to_lazy()


def test_case_11():
    tuple_0 = ()
    bool_0 = True
    maybe_0 = module_1.Maybe(tuple_0, bool_0)
    maybe_1 = module_1.Maybe(tuple_0, bool_0)
    var_0 = maybe_0.to_either()
    var_1 = maybe_1.bind(var_0)
    var_2 = var_1.to_lazy()
    var_3 = maybe_1.bind(var_0)
    var_4 = maybe_0.to_either()
    var_5 = var_2.to_either()
    var_6 = var_0.map(tuple_0)
    var_7 = maybe_0.to_lazy()


def test_case_12():
    none_type_0 = None
    str_0 = "@8A|v0>IDv"
    bool_0 = False
    maybe_0 = module_1.Maybe(str_0, bool_0)
    var_0 = maybe_0.to_box()
    bool_1 = var_0.__eq__(none_type_0)


def test_case_13():
    bytes_0 = b"\xe6u\x95o\xb4D\xb2\xf61\x934\xf9\xfc\xbc\xe9\x92\xa0x*\xf7"
    set_0 = {bytes_0}
    bool_0 = True
    maybe_0 = module_1.Maybe(bytes_0, bool_0)
    var_0 = maybe_0.to_lazy()
    bool_1 = False
    maybe_1 = module_1.Maybe(set_0, bool_1)
    var_1 = maybe_1.to_lazy()
    var_2 = var_0.to_box()


def test_case_14():
    int_0 = -770
    bool_0 = False
    bool_1 = False
    maybe_0 = module_1.Maybe(bool_0, bool_1)
    var_0 = maybe_0.to_lazy()
    var_1 = var_0.to_try()
    var_2 = maybe_0.to_either()
    var_2.get_or_else(int_0)


def test_case_15():
    bytes_0 = b"\x9a\xfc.\xad\xfb\xff "
    maybe_0 = module_1.Maybe(bytes_0, bytes_0)
    var_0 = maybe_0.bind(bytes_0)
    var_1 = maybe_0.to_either()
    var_2 = maybe_0.filter(bytes_0)
    var_3 = var_2.to_validation()
    var_4 = var_1.bind(var_1)
    var_5 = var_2.to_lazy()
    var_6 = maybe_0.map(var_1)
    var_7 = var_0.to_box()
    var_8 = var_6.to_try()
    var_9 = var_6.bind(var_8)
    var_10 = var_5.to_box()


def test_case_16():
    none_type_0 = None
    bool_0 = False
    maybe_0 = module_1.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.to_box()
    var_1 = maybe_0.to_try()
    var_0.bind(var_0)


def test_case_17():
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_1.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.to_validation()
    var_1 = maybe_0.to_box()
    var_2 = maybe_0.to_box()
    var_3 = var_0.to_box()
    var_2.get_or_else(bool_0)


def test_case_18():
    none_type_0 = None
    bool_0 = False
    maybe_0 = module_1.Maybe(bool_0, none_type_0)
    var_0 = maybe_0.to_either()
    maybe_1 = module_1.Maybe(none_type_0, bool_0)
    var_1 = maybe_0.to_either()
    bool_1 = maybe_1.__eq__(none_type_0)
    bool_2 = maybe_1.__eq__(maybe_1)
    maybe_1.ap(var_0)


def test_case_19():
    bytes_0 = b"\x9a\xfc.\xad\xfb\xff "
    maybe_0 = module_1.Maybe(bytes_0, bytes_0)
    var_0 = maybe_0.bind(bytes_0)
    var_1 = maybe_0.to_box()
    var_2 = maybe_0.filter(var_1)
    bool_0 = maybe_0.__eq__(var_0)
    var_3 = maybe_0.map(bytes_0)
    var_4 = var_1.to_validation()
    var_5 = maybe_0.to_try()
    var_6 = var_3.filter(var_5)
    bool_1 = var_5.__eq__(var_6)
    var_7 = maybe_0.get_or_else(var_4)
    var_8 = var_2.bind(var_5)
    var_9 = var_4.to_lazy()
    var_10 = maybe_0.map(var_7)
    bool_2 = False
    maybe_1 = module_1.Maybe(var_9, bool_2)
    var_11 = var_3.to_box()
    var_12 = maybe_1.to_either()
    var_13 = var_9.bind(bool_2)
    var_14 = var_10.to_lazy()
    var_13.get_or_else(var_4)
