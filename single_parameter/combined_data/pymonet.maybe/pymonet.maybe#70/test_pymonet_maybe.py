# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.maybe as module_0


def test_case_0():
    bytes_0 = b"\xee\x92j,\x9a\xa8\xbf;b\x1dCP\xe5a"
    maybe_0 = module_0.Maybe(bytes_0, bytes_0)


def test_case_1():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)


def test_case_2():
    str_0 = "8Y/ )A/cZl<azG[k"
    maybe_0 = module_0.Maybe(str_0, str_0)
    bool_0 = maybe_0.__eq__(maybe_0)
    var_0 = maybe_0.get_or_else(maybe_0)
    var_1 = var_0.map(var_0)
    var_2 = var_0.map(var_0)
    var_3 = var_1.to_validation()
    none_type_0 = None
    var_4 = var_1.filter(str_0)
    none_type_0.to_either()


def test_case_3():
    bool_0 = False
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    bool_1 = maybe_0.to_either()
    none_type_1 = None
    maybe_1 = module_0.Maybe(bool_0, none_type_1)
    bool_2 = maybe_1.__eq__(none_type_0)


def test_case_4():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.to_try()
    maybe_0.map(var_0)


def test_case_5():
    str_0 = "8Y/ )A/cZl<azG[k"
    maybe_0 = module_0.Maybe(str_0, str_0)
    bool_0 = maybe_0.__eq__(maybe_0)
    var_0 = maybe_0.get_or_else(maybe_0)
    var_1 = maybe_0.bind(var_0)
    var_2 = maybe_0.filter(var_0)
    var_3 = var_0.map(var_0)
    bool_1 = var_3.__eq__(var_2)
    var_4 = maybe_0.to_try()
    none_type_0 = None
    var_5 = var_4.bind(none_type_0)
    var_6 = var_2.to_lazy()
    var_7 = var_3.get_or_else(var_0)
    var_8 = var_7.to_either()


def test_case_6():
    bool_0 = True
    set_0 = {bool_0, bool_0}
    dict_0 = {}
    bool_1 = False
    maybe_0 = module_0.Maybe(dict_0, bool_1)
    maybe_0.bind(set_0)


def test_case_7():
    bytes_0 = b"lf0\x15S"
    bool_0 = False
    set_0 = {bool_0, bool_0, bool_0, bool_0}
    bool_1 = True
    none_type_0 = None
    bool_2 = True
    maybe_0 = module_0.Maybe(bool_1, bool_2)
    var_0 = maybe_0.filter(none_type_0)
    var_1 = var_0.to_either()
    var_2 = maybe_0.ap(set_0)
    var_3 = maybe_0.map(bytes_0)


def test_case_8():
    bytes_0 = b""
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    maybe_0.ap(bytes_0)


def test_case_9():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    bool_0 = True
    maybe_1 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_1.to_try()
    maybe_0.filter(var_0)


def test_case_10():
    str_0 = "C+}1~&y"
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.to_either()
    var_1 = maybe_0.get_or_else(str_0)
    var_1.to_box()


def test_case_11():
    bool_0 = False
    bool_1 = False
    maybe_0 = module_0.Maybe(bool_1, bool_1)
    var_0 = maybe_0.get_or_else(bool_0)
    var_0.to_validation()


def test_case_12():
    str_0 = "W:z,@MyY"
    maybe_0 = module_0.Maybe(str_0, str_0)
    none_type_0 = None
    var_0 = maybe_0.to_validation()
    var_1 = maybe_0.filter(var_0)
    var_2 = maybe_0.to_box()
    maybe_1 = module_0.Maybe(none_type_0, none_type_0)
    maybe_1.filter(maybe_0)


def test_case_13():
    none_type_0 = None
    bool_0 = False
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.to_box()


def test_case_14():
    str_0 = "8Y/ )A/cZl<azG[k"
    maybe_0 = module_0.Maybe(str_0, str_0)
    bool_0 = maybe_0.__eq__(maybe_0)
    var_0 = maybe_0.get_or_else(maybe_0)
    var_1 = var_0.map(var_0)
    var_2 = var_0.map(var_0)
    var_3 = maybe_0.to_try()
    var_4 = var_1.filter(var_3)
    var_5 = maybe_0.to_lazy()
    var_6 = var_5.to_either()


def test_case_15():
    bool_0 = False
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.to_lazy()
    none_type_1 = None
    maybe_1 = module_0.Maybe(bool_0, none_type_1)
    bool_1 = maybe_1.__eq__(none_type_0)


def test_case_16():
    int_0 = 1
    bool_0 = False
    maybe_0 = module_0.Maybe(int_0, bool_0)
    var_0 = maybe_0.to_try()
    var_0.to_validation()


def test_case_17():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.to_validation()
    var_0.to_validation()


def test_case_18():
    bytes_0 = b""
    maybe_0 = module_0.Maybe(bytes_0, bytes_0)
    bool_0 = maybe_0.__eq__(maybe_0)
    var_0 = maybe_0.to_validation()
    var_0.bind(bytes_0)


def test_case_19():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    bool_0 = maybe_0.__eq__(maybe_0)
    var_0 = maybe_0.to_lazy()
    var_1 = var_0.to_validation()
    maybe_0.bind(var_0)


def test_case_20():
    str_0 = "8Y/ )A/cZl<azG[k"
    maybe_0 = module_0.Maybe(str_0, str_0)
    bool_0 = maybe_0.__eq__(maybe_0)
    var_0 = maybe_0.get_or_else(maybe_0)
    var_1 = maybe_0.bind(var_0)
    var_2 = var_1.bind(var_0)
    var_3 = maybe_0.filter(var_0)
    var_4 = var_1.map(str_0)
    bool_1 = var_0.__eq__(var_1)
    var_5 = var_0.to_lazy()
    var_6 = var_5.to_try()
    var_7 = var_3.bind(var_0)
    var_8 = var_2.to_lazy()
    var_9 = var_5.map(bool_0)
    var_9.get_or_else(maybe_0)
