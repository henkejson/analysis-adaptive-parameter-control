# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.maybe as module_0


def test_case_0():
    int_0 = 2
    maybe_0 = module_0.Maybe(int_0, int_0)


def test_case_1():
    int_0 = 0
    maybe_0 = module_0.Maybe(int_0, int_0)


def test_case_2():
    bool_0 = False
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    bool_1 = maybe_0.__eq__(maybe_0)
    var_0 = maybe_0.to_try()
    bool_2 = True
    maybe_1 = module_0.Maybe(bool_0, bool_2)
    var_1 = maybe_1.filter(var_0)
    var_2 = maybe_1.filter(bool_0)
    var_3 = var_2.ap(var_0)
    maybe_0.filter(var_3)


def test_case_3():
    int_0 = -3816
    maybe_0 = module_0.Maybe(int_0, int_0)
    var_0 = maybe_0.get_or_else(maybe_0)
    var_1 = var_0.filter(var_0)
    var_2 = maybe_0.to_either()
    var_3 = var_1.to_lazy()
    bool_0 = maybe_0.__eq__(var_3)


def test_case_4():
    float_0 = -911.219835
    str_0 = "qJrU< 5/iV9Z&663X"
    none_type_0 = None
    bool_0 = False
    bool_1 = True
    maybe_0 = module_0.Maybe(none_type_0, str_0)
    var_0 = maybe_0.ap(bool_1)
    var_1 = var_0.to_either()
    var_2 = var_1.bind(bool_0)
    bool_2 = True
    maybe_1 = module_0.Maybe(none_type_0, bool_2)
    var_3 = maybe_1.map(str_0)
    var_4 = var_3.to_validation()
    var_5 = var_4.to_lazy()
    var_6 = var_5.ap(float_0)


def test_case_5():
    float_0 = 513.368
    bool_0 = False
    maybe_0 = module_0.Maybe(float_0, bool_0)
    var_0 = maybe_0.to_box()
    var_1 = maybe_0.to_try()
    bool_1 = var_1.__eq__(bool_0)
    bool_2 = maybe_0.__eq__(var_0)
    maybe_0.map(maybe_0)


def test_case_6():
    bool_0 = True
    bool_1 = True
    maybe_0 = module_0.Maybe(bool_0, bool_1)
    var_0 = maybe_0.get_or_else(bool_0)
    maybe_1 = module_0.Maybe(bool_0, bool_0)
    var_1 = maybe_1.to_box()
    bytes_0 = b"\x06\xa1c'\x08o\xb0\xdc\xbd\xaa\x172K\x8d\x9e\xc7\xffo"
    maybe_2 = module_0.Maybe(bytes_0, bytes_0)
    var_2 = maybe_2.to_validation()
    var_3 = maybe_2.to_validation()
    var_4 = maybe_2.bind(bytes_0)
    var_5 = var_4.to_validation()
    var_6 = maybe_2.to_validation()
    bool_2 = maybe_2.__eq__(maybe_2)
    var_7 = var_5.to_try()
    var_8 = var_6.to_try()
    bool_3 = var_0.__eq__(var_6)


def test_case_7():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.to_lazy()
    maybe_0.bind(none_type_0)


def test_case_8():
    bool_0 = False
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.to_try()
    maybe_1 = module_0.Maybe(bool_0, maybe_0)
    var_1 = maybe_1.filter(var_0)
    var_2 = maybe_1.filter(bool_0)
    var_3 = var_2.ap(var_0)
    maybe_0.filter(var_3)


def test_case_9():
    bool_0 = False
    bool_1 = True
    none_type_0 = None
    maybe_0 = module_0.Maybe(bool_0, none_type_0)
    maybe_0.ap(bool_1)


def test_case_10():
    bool_0 = False
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    maybe_1 = module_0.Maybe(bool_0, maybe_0)
    var_0 = maybe_1.filter(bool_0)
    maybe_0.filter(var_0)


def test_case_11():
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.get_or_else(none_type_0)
    none_type_1 = None
    bool_1 = False
    maybe_1 = module_0.Maybe(none_type_1, bool_1)
    var_1 = maybe_1.to_lazy()


def test_case_12():
    bool_0 = False
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.get_or_else(maybe_0)
    maybe_1 = module_0.Maybe(bool_0, maybe_0)
    var_1 = maybe_1.filter(var_0)
    maybe_0.filter(var_1)


def test_case_13():
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.to_either()


def test_case_14():
    bool_0 = False
    set_0 = {bool_0, bool_0}
    bool_1 = False
    tuple_0 = (set_0, bool_1)
    maybe_0 = module_0.Maybe(tuple_0, bool_0)
    none_type_0 = None
    str_0 = "b{"
    bool_2 = False
    maybe_1 = module_0.Maybe(str_0, bool_2)
    var_0 = maybe_1.to_either()
    var_0.ap(none_type_0)


def test_case_15():
    float_0 = 513.368
    bool_0 = True
    maybe_0 = module_0.Maybe(float_0, bool_0)
    var_0 = maybe_0.to_box()
    var_1 = maybe_0.to_try()
    var_2 = maybe_0.to_try()
    bool_1 = var_2.__eq__(var_1)
    var_3 = var_0.to_lazy()
    bool_2 = var_0.__eq__(maybe_0)


def test_case_16():
    int_0 = 0
    maybe_0 = module_0.Maybe(int_0, int_0)
    var_0 = maybe_0.to_box()


def test_case_17():
    str_0 = "u"
    maybe_0 = module_0.Maybe(str_0, str_0)
    var_0 = maybe_0.to_lazy()
    var_1 = maybe_0.to_lazy()


def test_case_18():
    bool_0 = False
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.to_try()
    maybe_1 = module_0.Maybe(bool_0, maybe_0)
    var_1 = maybe_1.filter(var_0)
    maybe_0.filter(var_1)


def test_case_19():
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.to_validation()
    maybe_1 = module_0.Maybe(maybe_0, bool_0)
    var_1 = maybe_1.to_either()


def test_case_20():
    bool_0 = False
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = module_0.Maybe(bool_0, bool_0)
    maybe_1 = module_0.Maybe(bool_0, maybe_0)
    var_1 = var_0.to_validation()
    var_2 = maybe_1.filter(bool_0)
    var_3 = var_2.ap(var_0)
    maybe_0.filter(var_3)


def test_case_21():
    int_0 = -758
    set_0 = {int_0, int_0, int_0}
    int_1 = 1632
    maybe_0 = module_0.Maybe(set_0, int_1)
    var_0 = maybe_0.to_lazy()
    var_1 = var_0.to_validation()
    var_2 = var_1.to_box()
    var_0.to_lazy()


def test_case_22():
    tuple_0 = ()
    float_0 = -1133.0
    maybe_0 = module_0.Maybe(float_0, tuple_0)
    var_0 = maybe_0.to_lazy()
    var_1 = var_0.to_box()
    var_2 = var_1.to_try()
    var_2.to_validation()


def test_case_23():
    float_0 = -2107.65323
    maybe_0 = module_0.Maybe(float_0, float_0)
    none_type_0 = None
    var_0 = maybe_0.ap(none_type_0)
    var_1 = var_0.to_box()
    var_2 = var_1.to_validation()
    var_3 = maybe_0.to_box()
    var_4 = maybe_0.to_validation()
    var_5 = maybe_0.ap(maybe_0)
    var_6 = maybe_0.to_validation()
    var_7 = var_2.to_try()
    var_8 = maybe_0.map(var_6)
    var_9 = var_8.to_try()
    maybe_1 = module_0.Maybe(float_0, none_type_0)
    bool_0 = maybe_0.__eq__(maybe_1)
    var_10 = maybe_1.to_try()
    var_11 = maybe_1.to_either()
    var_12 = var_0.filter(var_2)
    bool_1 = var_11.__eq__(var_12)
    var_3.map(var_6)
