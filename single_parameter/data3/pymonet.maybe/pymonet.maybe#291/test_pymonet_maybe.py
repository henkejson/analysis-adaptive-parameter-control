# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.maybe as module_0
import typing as module_1
import builtins as module_2


def test_case_0():
    bytes_0 = b"&\xdb\xde\xf5l\x1d^"
    maybe_0 = module_0.Maybe(bytes_0, bytes_0)


def test_case_1():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)


def test_case_2():
    int_0 = 0
    maybe_0 = module_0.Maybe(int_0, int_0)
    bool_0 = maybe_0.__eq__(int_0)
    bool_1 = maybe_0.__eq__(maybe_0)
    var_0 = maybe_0.to_either()
    var_1 = maybe_0.to_either()
    var_2 = maybe_0.to_validation()
    var_3 = maybe_0.to_validation()


def test_case_3():
    int_0 = 0
    maybe_0 = module_0.Maybe(int_0, int_0)
    bool_0 = maybe_0.__eq__(int_0)
    var_0 = maybe_0.to_either()
    var_1 = maybe_0.to_validation()
    var_2 = maybe_0.to_validation()


def test_case_4():
    bytes_0 = b"\xd0\xf3\xd1\x84_"
    generic_0 = module_1.Generic()
    maybe_0 = module_0.Maybe(bytes_0, generic_0)
    bool_0 = maybe_0.__eq__(bytes_0)
    var_0 = maybe_0.to_try()
    bool_1 = True
    var_1 = maybe_0.get_or_else(bool_1)
    none_type_0 = None
    maybe_1 = module_0.Maybe(none_type_0, none_type_0)
    var_2 = maybe_1.to_box()
    var_3 = maybe_0.map(var_1)
    var_4 = var_2.to_try()
    bool_2 = maybe_1.__eq__(maybe_0)


def test_case_5():
    bool_0 = False
    dict_0 = {bool_0: bool_0}
    none_type_0 = None
    bool_1 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_1)
    var_0 = maybe_0.filter(bool_0)
    maybe_1 = module_0.Maybe(maybe_0, maybe_0)
    var_1 = maybe_0.to_validation()
    var_2 = maybe_0.filter(var_1)
    maybe_2 = module_0.Maybe(bool_1, bool_1)
    maybe_3 = module_0.Maybe(var_0, bool_0)
    var_3 = maybe_3.to_box()
    maybe_3.map(dict_0)


def test_case_6():
    bytes_0 = b"6\x0eP\xcaK\xbb#p\xdf$\xf1>"
    maybe_0 = module_0.Maybe(bytes_0, bytes_0)
    var_0 = maybe_0.ap(bytes_0)
    var_1 = var_0.map(var_0)
    var_2 = var_1.bind(var_0)
    var_3 = var_1.get_or_else(maybe_0)
    var_4 = maybe_0.to_either()
    var_5 = var_4.map(var_3)
    var_6 = maybe_0.filter(var_0)
    bool_0 = var_1.__eq__(var_0)
    var_7 = var_4.to_lazy()
    var_8 = maybe_0.to_box()
    var_9 = var_3.bind(var_0)
    maybe_1 = module_0.Maybe(var_8, var_0)
    var_10 = var_4.to_box()


def test_case_7():
    none_type_0 = None
    int_0 = -311
    set_0 = {int_0, int_0, int_0}
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.to_lazy()
    bool_0 = var_0.__eq__(int_0)
    maybe_1 = module_0.Maybe(set_0, set_0)
    var_1 = maybe_1.to_either()
    maybe_0.bind(set_0)


def test_case_8():
    bool_0 = False
    dict_0 = {bool_0: bool_0}
    none_type_0 = None
    bool_1 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_1)
    var_0 = maybe_0.to_validation()
    maybe_1 = module_0.Maybe(maybe_0, maybe_0)
    var_1 = maybe_0.to_validation()
    var_2 = maybe_0.filter(var_1)
    var_3 = var_2.to_lazy()
    var_4 = maybe_1.ap(dict_0)
    var_5 = var_3.bind(var_4)
    var_1.to_validation()


def test_case_9():
    bool_0 = False
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    bool_1 = maybe_0.__eq__(none_type_0)
    maybe_0.ap(bool_0)


def test_case_10():
    object_0 = module_2.object()
    bool_0 = False
    maybe_0 = module_0.Maybe(object_0, bool_0)
    maybe_0.filter(bool_0)


def test_case_11():
    bytes_0 = b"&\xdb\xde\xf5l\x1d^"
    bool_0 = True
    maybe_0 = module_0.Maybe(bytes_0, bool_0)
    none_type_0 = None
    var_0 = maybe_0.get_or_else(none_type_0)


def test_case_12():
    float_0 = 3886.598
    none_type_0 = None
    maybe_0 = module_0.Maybe(float_0, none_type_0)
    var_0 = maybe_0.get_or_else(float_0)
    var_0.to_either()


def test_case_13():
    str_0 = "Jk{k1%n>b0"
    maybe_0 = module_0.Maybe(str_0, str_0)
    var_0 = maybe_0.to_either()
    var_1 = maybe_0.to_either()
    var_0.get_or_else(var_0)


def test_case_14():
    str_0 = "&J?Sq-?m_\\+y"
    none_type_0 = None
    maybe_0 = module_0.Maybe(str_0, none_type_0)
    var_0 = maybe_0.to_either()
    var_1 = var_0.to_box()


def test_case_15():
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.to_box()
    bool_1 = maybe_0.__eq__(none_type_0)
    module_1.Generic(*var_0)


def test_case_16():
    object_0 = module_2.object()
    bool_0 = False
    maybe_0 = module_0.Maybe(object_0, bool_0)
    var_0 = maybe_0.to_box()
    var_1 = maybe_0.to_box()


def test_case_17():
    bool_0 = True
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    tuple_0 = (bool_0, bool_0, bool_0)
    var_0 = maybe_0.get_or_else(tuple_0)
    var_1 = maybe_0.to_lazy()
    var_2 = var_1.to_try()
    maybe_1 = module_0.Maybe(tuple_0, tuple_0)
    maybe_2 = module_0.Maybe(tuple_0, bool_0)


def test_case_18():
    int_0 = 0
    maybe_0 = module_0.Maybe(int_0, int_0)
    bool_0 = maybe_0.to_lazy()
    var_0 = maybe_0.to_either()
    var_1 = maybe_0.to_lazy()
    maybe_1 = module_0.Maybe(maybe_0, var_1)
    var_2 = maybe_0.to_either()
    bool_1 = maybe_0.__eq__(int_0)
    var_3 = maybe_1.get_or_else(int_0)
    var_4 = maybe_1.filter(var_3)
    var_0.map(bool_0)


def test_case_19():
    bytes_0 = b"\xd7\nM\xdc\xce\x9d\xbf\xe60A\xa2\xc9\x82\xef"
    bool_0 = False
    bool_1 = False
    maybe_0 = module_0.Maybe(bytes_0, bool_1)
    maybe_1 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_1.to_try()
    maybe_1.filter(var_0)


def test_case_20():
    bytes_0 = b"\xd0\xf3\xd1\x84_"
    generic_0 = module_1.Generic()
    maybe_0 = module_0.Maybe(bytes_0, generic_0)
    bool_0 = maybe_0.__eq__(bytes_0)
    bool_1 = True
    var_0 = maybe_0.get_or_else(bool_1)
    none_type_0 = None
    maybe_1 = module_0.Maybe(none_type_0, none_type_0)
    var_1 = maybe_1.to_box()
    var_2 = maybe_0.map(var_0)
    var_3 = var_2.to_either()
    var_4 = maybe_1.to_validation()
    var_5 = maybe_0.to_validation()


def test_case_21():
    str_0 = "GHwD/*j:+"
    set_0 = {str_0, str_0, str_0}
    bool_0 = False
    maybe_0 = module_0.Maybe(set_0, bool_0)
    var_0 = maybe_0.to_validation()


def test_case_22():
    str_0 = 'K`u$%,"u~A?I0*CLiShC'
    str_0.to_box()


def test_case_23():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.to_lazy()
    var_1 = var_0.to_either()
    complex_0 = -428.47 - 3213.961j
    maybe_1 = module_0.Maybe(complex_0, complex_0)
    var_2 = maybe_1.to_lazy()
    var_3 = maybe_1.map(maybe_1)
    var_4 = maybe_1.to_either()
    var_5 = maybe_1.filter(var_2)
    var_6 = maybe_1.to_either()
    var_7 = maybe_1.to_validation()
    var_4.to_either()
