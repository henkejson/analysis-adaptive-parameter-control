# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.maybe as module_0


def test_case_0():
    str_0 = "]hFa}lXTBjfU:=}"
    maybe_0 = module_0.Maybe(str_0, str_0)


def test_case_1():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)


def test_case_2():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.bind(maybe_0)
    var_1 = maybe_0.to_try()
    bool_1 = maybe_0.__eq__(var_0)
    bool_2 = maybe_0.__eq__(var_0)


def test_case_3():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.to_box()
    bool_1 = maybe_0.__eq__(var_0)


def test_case_4():
    none_type_0 = None
    bytes_0 = b"\xf2"
    set_0 = {bytes_0, bytes_0, bytes_0, bytes_0}
    list_0 = [set_0, bytes_0, bytes_0, bytes_0]
    maybe_0 = module_0.Maybe(list_0, list_0)
    var_0 = maybe_0.map(none_type_0)
    var_1 = var_0.to_try()


def test_case_5():
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.ap(none_type_0)
    var_1 = var_0.filter(var_0)
    maybe_1 = module_0.Maybe(var_1, var_0)
    maybe_2 = module_0.Maybe(bool_0, none_type_0)
    var_2 = var_0.filter(maybe_2)
    var_3 = maybe_0.to_box()
    maybe_2.map(var_3)


def test_case_6():
    bool_0 = False
    none_type_0 = None
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    maybe_0.bind(none_type_0)


def test_case_7():
    none_type_0 = None
    int_0 = 0
    bool_0 = True
    maybe_0 = module_0.Maybe(int_0, bool_0)
    var_0 = maybe_0.ap(none_type_0)
    var_1 = var_0.to_box()


def test_case_8():
    none_type_0 = None
    list_0 = []
    maybe_0 = module_0.Maybe(list_0, list_0)
    maybe_0.ap(none_type_0)


def test_case_9():
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.ap(none_type_0)
    maybe_1 = module_0.Maybe(var_0, var_0)
    var_1 = maybe_1.to_either()
    var_2 = maybe_1.to_validation()
    var_3 = maybe_1.filter(maybe_1)
    bool_1 = False
    bool_2 = True
    maybe_2 = module_0.Maybe(maybe_1, var_0)
    var_4 = var_0.to_try()
    var_5 = var_3.bind(bool_2)
    maybe_3 = module_0.Maybe(maybe_0, bool_1)
    maybe_3.filter(var_1)


def test_case_10():
    bool_0 = False
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.to_box()
    bool_1 = maybe_0.__eq__(var_0)
    maybe_0.filter(bool_0)


def test_case_11():
    float_0 = 3970.38148
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.to_box()
    var_1 = maybe_0.get_or_else(float_0)
    var_1.filter(none_type_0)


def test_case_12():
    float_0 = 60.1
    dict_0 = {}
    maybe_0 = module_0.Maybe(dict_0, dict_0)
    var_0 = maybe_0.to_validation()
    var_1 = maybe_0.get_or_else(float_0)
    var_2 = maybe_0.to_try()
    bytes_0 = b"\\\xcc"
    bool_0 = True
    maybe_1 = module_0.Maybe(bytes_0, bool_0)
    var_2.to_box()


def test_case_13():
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.ap(none_type_0)
    var_1 = var_0.to_try()
    maybe_1 = module_0.Maybe(var_1, var_0)
    maybe_2 = module_0.Maybe(bool_0, none_type_0)
    var_2 = var_0.filter(maybe_2)
    var_3 = maybe_2.to_either()
    var_4 = maybe_0.to_box()
    maybe_2.map(var_4)


def test_case_14():
    bytes_0 = b"p\x98qx"
    maybe_0 = module_0.Maybe(bytes_0, bytes_0)
    var_0 = maybe_0.to_box()


def test_case_15():
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.ap(none_type_0)
    maybe_1 = module_0.Maybe(var_0, var_0)
    var_1 = var_0.map(var_0)
    var_2 = maybe_1.to_validation()
    var_3 = maybe_1.filter(maybe_1)
    var_4 = var_1.to_lazy()
    bool_1 = False
    bool_2 = True
    maybe_2 = module_0.Maybe(maybe_1, var_0)
    var_5 = var_0.to_try()
    var_6 = var_3.bind(bool_2)
    maybe_3 = module_0.Maybe(maybe_0, bool_1)
    maybe_3.filter(var_1)


def test_case_16():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.to_lazy()
    var_1 = var_0.to_either()


def test_case_17():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.to_box()
    var_1 = maybe_0.to_try()
    bool_1 = maybe_0.__eq__(var_0)
    bool_2 = maybe_0.__eq__(var_0)


def test_case_18():
    none_type_0 = None
    bool_0 = False
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.to_try()


def test_case_19():
    int_0 = -2898
    bool_0 = False
    str_0 = "U"
    str_1 = "Max[value={}]"
    list_0 = [bool_0, str_0, bool_0, str_1]
    maybe_0 = module_0.Maybe(bool_0, list_0)
    var_0 = maybe_0.map(int_0)
    var_1 = var_0.to_validation()


def test_case_20():
    bool_0 = True
    set_0 = {bool_0, bool_0, bool_0, bool_0}
    none_type_0 = None
    bool_1 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_1)
    var_0 = maybe_0.map(set_0)
    var_1 = var_0.to_validation()
    none_type_1 = None
    tuple_0 = (bool_0, bool_0, var_1, set_0)
    var_2 = var_0.ap(tuple_0)
    var_3 = maybe_0.ap(none_type_0)
    maybe_1 = module_0.Maybe(none_type_1, var_0)
    var_4 = maybe_1.to_box()
    var_5 = var_4.ap(var_3)
    var_6 = var_0.filter(var_4)
    var_7 = var_2.to_lazy()
    var_8 = var_7.to_validation()
    var_9 = var_7.bind(var_1)
    var_10 = maybe_1.to_lazy()
    var_1.to_validation()


def test_case_21():
    none_type_0 = None
    bytes_0 = b"\x9c\xf9\x84%#L\x92\x9d\xca<L\xb6\xc0"
    maybe_0 = module_0.Maybe(bytes_0, none_type_0)
    bool_0 = maybe_0.__eq__(maybe_0)
    maybe_0.filter(bool_0)


def test_case_22():
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.ap(none_type_0)
    var_1 = var_0.to_try()
    maybe_1 = module_0.Maybe(var_1, var_0)
    var_2 = maybe_1.ap(var_1)
    var_3 = var_0.filter(var_2)
    var_4 = var_2.to_either()
    var_5 = maybe_0.to_box()
    var_6 = var_2.map(var_5)
    var_7 = maybe_0.to_validation()
    bool_1 = maybe_0.__eq__(var_2)
    var_8 = var_3.filter(bool_1)
    maybe_2 = module_0.Maybe(var_3, maybe_0)
    maybe_3 = module_0.Maybe(none_type_0, var_7)
    var_9 = maybe_2.ap(var_2)
    var_10 = maybe_1.to_either()
    var_11 = maybe_0.to_box()
    var_12 = var_11.to_try()
    var_13 = var_4.bind(var_4)
    bool_2 = maybe_3.__eq__(maybe_2)
    bool_3 = maybe_2.__eq__(var_2)
