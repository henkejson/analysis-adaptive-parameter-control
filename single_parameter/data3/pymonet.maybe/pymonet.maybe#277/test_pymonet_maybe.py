# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.maybe as module_0


def test_case_0():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)


def test_case_1():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)


def test_case_2():
    bool_0 = True
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    maybe_1 = module_0.Maybe(maybe_0, none_type_0)
    var_0 = maybe_1.to_box()
    bool_1 = maybe_1.__eq__(maybe_0)
    var_0.get_or_else(bool_0)


def test_case_3():
    list_0 = []
    int_0 = 2919
    maybe_0 = module_0.Maybe(int_0, int_0)
    var_0 = maybe_0.map(list_0)
    var_1 = var_0.to_either()


def test_case_4():
    float_0 = -512.526193
    none_type_0 = None
    maybe_0 = module_0.Maybe(float_0, none_type_0)
    var_0 = maybe_0.to_box()
    maybe_0.map(var_0)


def test_case_5():
    bytes_0 = b"6%\xbdQ\xd1\x07\x19bC\xdd\xea"
    maybe_0 = module_0.Maybe(bytes_0, bytes_0)
    var_0 = maybe_0.bind(bytes_0)
    var_1 = var_0.filter(maybe_0)


def test_case_6():
    int_0 = -2125
    str_0 = "qzy<"
    dict_0 = {}
    list_0 = [dict_0, dict_0, dict_0, dict_0]
    tuple_0 = (str_0, dict_0, str_0, list_0)
    none_type_0 = None
    maybe_0 = module_0.Maybe(tuple_0, none_type_0)
    maybe_0.bind(int_0)


def test_case_7():
    bool_0 = True
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.ap(bool_0)
    var_1 = var_0.filter(none_type_0)
    var_2 = var_1.to_try()
    bool_1 = False
    maybe_1 = module_0.Maybe(bool_0, bool_1)
    maybe_1.filter(var_2)


def test_case_8():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    maybe_0.ap(none_type_0)


def test_case_9():
    int_0 = 0
    maybe_0 = module_0.Maybe(int_0, int_0)
    maybe_0.filter(int_0)


def test_case_10():
    bool_0 = True
    bytes_0 = b"+.\xa4\x00\x03=\x04N"
    maybe_0 = module_0.Maybe(bytes_0, bool_0)
    var_0 = maybe_0.to_lazy()
    var_1 = maybe_0.filter(var_0)
    tuple_0 = ()
    bool_1 = True
    var_2 = maybe_0.to_either()
    var_3 = maybe_0.bind(var_2)
    maybe_1 = module_0.Maybe(tuple_0, bool_1)
    var_4 = maybe_1.to_box()
    var_5 = var_4.to_either()
    var_6 = var_4.to_try()
    var_7 = var_5.to_try()
    var_8 = maybe_0.to_validation()
    var_9 = var_0.to_box()
    var_10 = maybe_1.get_or_else(var_4)
    var_2.get_or_else(var_8)


def test_case_11():
    int_0 = 0
    maybe_0 = module_0.Maybe(int_0, int_0)
    var_0 = maybe_0.get_or_else(int_0)
    maybe_0.filter(int_0)


def test_case_12():
    int_0 = 0
    maybe_0 = module_0.Maybe(int_0, int_0)
    var_0 = maybe_0.to_either()
    maybe_0.filter(int_0)


def test_case_13():
    bytes_0 = b"6%\xbdQ\xd1\x07\x19bC\xdd\xea"
    maybe_0 = module_0.Maybe(bytes_0, bytes_0)
    var_0 = maybe_0.to_box()


def test_case_14():
    int_0 = 0
    maybe_0 = module_0.Maybe(int_0, int_0)
    var_0 = maybe_0.to_box()
    maybe_0.filter(var_0)


def test_case_15():
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.to_lazy()
    var_1 = var_0.bind(none_type_0)
    var_1.to_validation()


def test_case_16():
    int_0 = 0
    maybe_0 = module_0.Maybe(int_0, int_0)
    var_0 = maybe_0.to_lazy()
    maybe_0.filter(var_0)


def test_case_17():
    bytes_0 = b"\xbd\x95% \xf8QV\x16\xc3\xf9\xb2\xad\xd2\xcc\xde\xd0\x1e>"
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.to_try()
    var_1 = maybe_0.get_or_else(bytes_0)
    var_1.to_lazy()


def test_case_18():
    bool_0 = True
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.ap(bool_0)
    var_1 = var_0.filter(none_type_0)
    var_2 = var_0.to_validation()
    var_3 = var_1.to_try()
    bool_1 = False
    maybe_1 = module_0.Maybe(bool_0, bool_1)
    maybe_1.filter(var_3)


def test_case_19():
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    dict_0 = {}
    maybe_1 = module_0.Maybe(maybe_0, dict_0)
    var_0 = maybe_1.to_validation()


def test_case_20():
    bool_0 = True
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.filter(none_type_0)
    var_1 = var_0.to_try()
    maybe_1 = module_0.Maybe(bool_0, none_type_0)
    maybe_1.filter(var_1)


def test_case_21():
    bool_0 = True
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.ap(bool_0)
    var_1 = var_0.filter(none_type_0)
    var_2 = var_1.to_try()
    bool_1 = False
    bool_2 = var_0.__eq__(var_0)
    var_3 = var_2.get_or_else(var_1)
    maybe_1 = module_0.Maybe(bool_0, bool_1)
    maybe_1.filter(var_2)


def test_case_22():
    bytes_0 = b"\x0fpn\xde!2\x1c2xI"
    int_0 = -1060
    bool_0 = False
    maybe_0 = module_0.Maybe(bytes_0, bool_0)
    var_0 = maybe_0.to_box()
    bool_1 = maybe_0.__eq__(var_0)
    var_1 = maybe_0.to_lazy()
    str_0 = "YQbpF$WN"
    dict_0 = {int_0: int_0, str_0: int_0}
    bool_2 = True
    maybe_1 = module_0.Maybe(dict_0, bool_2)
    var_2 = maybe_1.to_validation()
    maybe_2 = module_0.Maybe(var_0, var_0)
    var_3 = var_1.to_either()
    var_4 = maybe_1.to_validation()


def test_case_23():
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    bool_1 = False
    maybe_1 = module_0.Maybe(none_type_0, bool_1)
    var_0 = maybe_0.to_validation()
    var_1 = maybe_0.filter(var_0)
    var_2 = maybe_0.to_either()
    bool_2 = maybe_0.__eq__(maybe_1)
    var_3 = maybe_0.to_box()
    var_4 = var_2.to_lazy()
    var_5 = maybe_0.to_box()
    maybe_1.filter(bool_1)
