# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.maybe as module_0
import typing as module_1


def test_case_0():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)


def test_case_1():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)


def test_case_2():
    bool_0 = False
    list_0 = [bool_0]
    maybe_0 = module_0.Maybe(list_0, bool_0)
    var_0 = maybe_0.to_validation()
    var_1 = maybe_0.to_try()
    bool_1 = maybe_0.__eq__(var_1)
    var_1.to_try()


def test_case_3():
    bool_0 = True
    list_0 = [bool_0]
    maybe_0 = module_0.Maybe(list_0, bool_0)
    var_0 = maybe_0.to_validation()
    var_1 = maybe_0.to_try()
    var_2 = maybe_0.map(maybe_0)
    bool_1 = maybe_0.__eq__(var_1)
    var_3 = var_2.to_try()
    var_4 = maybe_0.to_lazy()
    var_4.filter(maybe_0)


def test_case_4():
    str_0 = "\n    Min is a Monoid that will combines 2 numbers, resulting in the smallest of the two.\n    "
    str_1 = "FIiF{@J6Gq)[$6S+q7"
    str_2 = "\t2w-1a_g6bj&\\YCx1'"
    maybe_0 = module_0.Maybe(str_1, str_2)
    none_type_0 = None
    var_0 = maybe_0.map(str_1)
    maybe_1 = module_0.Maybe(maybe_0, none_type_0)
    var_1 = maybe_1.get_or_else(maybe_1)
    maybe_1.map(str_0)


def test_case_5():
    bytes_0 = b"\x18@VqAO\x80\xa1"
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(bytes_0, bool_0)
    var_0 = maybe_0.to_lazy()
    var_1 = maybe_0.get_or_else(bytes_0)
    var_2 = maybe_0.to_box()
    int_0 = 2440
    maybe_1 = module_0.Maybe(none_type_0, int_0)
    var_3 = maybe_1.bind(bytes_0)
    var_4 = var_3.to_try()
    var_4.to_validation()


def test_case_6():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    maybe_0.bind(none_type_0)


def test_case_7():
    bytes_0 = b""
    bool_0 = True
    maybe_0 = module_0.Maybe(bytes_0, bool_0)
    var_0 = maybe_0.to_validation()
    none_type_0 = None
    var_1 = maybe_0.ap(none_type_0)
    var_2 = var_1.filter(var_0)


def test_case_8():
    bool_0 = False
    bool_1 = False
    maybe_0 = module_0.Maybe(bool_1, bool_1)
    maybe_0.ap(bool_0)


def test_case_9():
    bytes_0 = b"lf\x07\x02\x91 \xa0H\xb4! "
    list_0 = [bytes_0, bytes_0, bytes_0]
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    maybe_0.filter(list_0)


def test_case_10():
    int_0 = 1474
    maybe_0 = module_0.Maybe(int_0, int_0)
    bool_0 = False
    bool_1 = True
    maybe_1 = module_0.Maybe(int_0, bool_1)
    var_0 = maybe_1.get_or_else(bool_0)
    var_1 = maybe_1.filter(bool_0)
    maybe_2 = module_0.Maybe(int_0, bool_0)
    var_2 = maybe_2.to_validation()
    maybe_2.filter(bool_0)


def test_case_11():
    bool_0 = False
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.get_or_else(bool_0)
    var_0.get_or_else(bool_0)


def test_case_12():
    str_0 = "O_WC88bp!"
    bool_0 = True
    maybe_0 = module_0.Maybe(str_0, bool_0)
    var_0 = maybe_0.to_either()


def test_case_13():
    none_type_0 = None
    bool_0 = False
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.to_either()


def test_case_14():
    bytes_0 = b"\xb2\xbf r\xbb\xb8\xbd$\xf8wY\x92"
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.to_try()
    var_1 = var_0.map(none_type_0)
    var_2 = maybe_0.to_box()
    var_3 = var_2.to_try()
    maybe_1 = module_0.Maybe(bytes_0, bytes_0)
    var_2.bind(var_3)


def test_case_15():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    bool_0 = True
    var_0 = maybe_0.to_validation()
    maybe_1 = module_0.Maybe(none_type_0, bool_0)
    var_1 = maybe_0.to_lazy()
    float_0 = 203.754406
    var_2 = maybe_0.get_or_else(float_0)
    bool_1 = False
    maybe_2 = module_0.Maybe(var_1, bool_1)
    var_3 = maybe_0.to_box()
    var_2.filter(var_1)


def test_case_16():
    bytes_0 = b""
    bool_0 = True
    maybe_0 = module_0.Maybe(bytes_0, bool_0)
    var_0 = maybe_0.to_validation()
    none_type_0 = None
    var_1 = maybe_0.ap(none_type_0)
    var_2 = var_1.filter(var_0)
    var_3 = maybe_0.to_lazy()


def test_case_17():
    bool_0 = False
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.to_validation()
    var_1 = maybe_0.to_lazy()
    var_2 = maybe_0.get_or_else(bool_0)
    var_2.to_lazy()


def test_case_18():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.to_try()
    var_1 = maybe_0.filter(maybe_0)
    var_0.to_validation()


def test_case_19():
    none_type_0 = None
    bool_0 = False
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.to_try()
    var_0.to_either()


def test_case_20():
    int_0 = 1450
    maybe_0 = module_0.Maybe(int_0, int_0)
    bool_0 = False
    var_0 = maybe_0.to_either()
    var_1 = maybe_0.filter(bool_0)
    maybe_1 = module_0.Maybe(int_0, bool_0)
    var_2 = maybe_1.to_validation()
    maybe_1.filter(bool_0)


def test_case_21():
    none_type_0 = None
    none_type_0.to_box()


def test_case_22():
    generic_0 = module_1.Generic()
    bool_0 = False
    bool_1 = True
    set_0 = {bool_1, bool_1, bool_1}
    maybe_0 = module_0.Maybe(set_0, bool_1)
    var_0 = maybe_0.to_lazy()
    var_1 = var_0.to_try()
    var_1.map(bool_0)


def test_case_23():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.to_lazy()
    var_1 = var_0.to_try()
    var_2 = var_0.bind(none_type_0)
    var_1.ap(none_type_0)


def test_case_24():
    bool_0 = False
    bytes_0 = b"\x16\r\xb8\x95\xbe\xa1fJ\x8e\xf2\xc0\xb7\x88\xc8\xcd\xfb\x13\x15w"
    bool_1 = True
    maybe_0 = module_0.Maybe(bytes_0, bool_1)
    var_0 = maybe_0.ap(bool_0)
    var_1 = maybe_0.to_validation()
    maybe_1 = module_0.Maybe(bool_0, bool_0)
    bool_2 = maybe_1.__eq__(maybe_1)
    var_2 = var_1.to_try()
    var_2.to_lazy()


def test_case_25():
    bool_0 = False
    bytes_0 = b"\x16\r\xb8\x95\xbe\xa1fJ\x8e\xf2\xc0\xb7\x88\xc8\xcd\xfb\x13\x15w"
    bool_1 = True
    maybe_0 = module_0.Maybe(bytes_0, bool_1)
    var_0 = maybe_0.ap(bool_0)
    var_1 = maybe_0.to_validation()
    maybe_1 = module_0.Maybe(bool_0, bool_0)
    bool_2 = maybe_1.__eq__(maybe_1)
    bool_3 = var_0.__eq__(maybe_0)
    var_2 = var_0.to_either()
    var_3 = var_0.to_lazy()
    var_4 = maybe_0.to_either()
    bool_4 = var_3.__eq__(var_4)
    var_5 = var_4.map(var_4)
    var_6 = var_0.filter(bytes_0)
    var_7 = var_6.get_or_else(var_0)
    var_8 = var_7.to_lazy()
    var_9 = var_0.filter(maybe_1)
    var_10 = var_8.to_try()
    var_11 = var_6.to_either()
    maybe_2 = module_0.Maybe(bool_0, var_1)
    var_12 = maybe_2.ap(bool_4)
    var_13 = var_5.bind(bool_3)


def test_case_26():
    bool_0 = False
    bytes_0 = b"\x16\r\xb8\x95\xbe\xa1fJ\x8e\xf2\xc0\xb7\x88\xc8\xcd\xfb\x13\x15w"
    bool_1 = True
    maybe_0 = module_0.Maybe(bytes_0, bool_1)
    var_0 = maybe_0.ap(bool_0)
    maybe_1 = module_0.Maybe(bool_0, bool_0)
    bool_2 = maybe_1.__eq__(maybe_1)
    maybe_2 = module_0.Maybe(bool_1, var_0)
    var_1 = var_0.filter(maybe_0)
    bool_3 = maybe_2.__eq__(maybe_0)
    var_2 = var_1.to_either()
    var_3 = maybe_2.to_lazy()
    var_4 = maybe_2.to_either()
    bool_4 = var_3.__eq__(maybe_2)
    var_5 = var_4.map(var_4)
    var_6 = var_0.filter(bytes_0)
    maybe_1.filter(var_0)
