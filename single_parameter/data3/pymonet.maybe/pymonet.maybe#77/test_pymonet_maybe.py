# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.maybe as module_0
import builtins as module_1


def test_case_0():
    float_0 = 4536.057
    maybe_0 = module_0.Maybe(float_0, float_0)


def test_case_1():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)


def test_case_2():
    none_type_0 = None
    bool_0 = True
    bytes_0 = b"+\xac\xd1\x0e\x0e"
    maybe_0 = module_0.Maybe(bool_0, bytes_0)
    var_0 = maybe_0.filter(none_type_0)
    bool_1 = maybe_0.__eq__(var_0)
    var_1 = var_0.ap(var_0)
    var_2 = var_0.filter(var_1)
    var_3 = var_1.to_lazy()
    var_4 = var_3.to_box()
    var_5 = var_0.to_try()


def test_case_3():
    int_0 = -533
    maybe_0 = module_0.Maybe(int_0, int_0)
    var_0 = maybe_0.to_lazy()
    var_1 = maybe_0.to_either()
    bool_0 = True
    maybe_1 = module_0.Maybe(bool_0, bool_0)
    none_type_0 = None
    var_2 = maybe_1.filter(none_type_0)
    bool_1 = maybe_0.__eq__(int_0)
    var_3 = maybe_1.to_box()


def test_case_4():
    int_0 = -533
    maybe_0 = module_0.Maybe(int_0, int_0)
    var_0 = maybe_0.to_lazy()
    var_1 = maybe_0.to_either()
    bool_0 = True
    maybe_1 = module_0.Maybe(bool_0, bool_0)
    none_type_0 = None
    var_2 = maybe_1.filter(none_type_0)
    var_3 = maybe_1.to_validation()
    var_4 = var_1.to_lazy()
    var_5 = var_2.map(var_4)
    var_6 = var_5.to_lazy()


def test_case_5():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.to_box()
    maybe_0.map(var_0)


def test_case_6():
    int_0 = 976
    bool_0 = False
    maybe_0 = module_0.Maybe(int_0, int_0)
    var_0 = maybe_0.to_validation()
    var_1 = var_0.to_lazy()
    var_2 = var_1.to_validation()
    var_3 = maybe_0.bind(int_0)
    var_4 = var_0.to_lazy()
    dict_0 = {bool_0: bool_0}
    maybe_1 = module_0.Maybe(dict_0, bool_0)
    maybe_1.filter(int_0)


def test_case_7():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    maybe_1 = module_0.Maybe(none_type_0, none_type_0)
    object_0 = module_1.object()
    bool_0 = maybe_0.__eq__(object_0)
    var_0 = maybe_1.to_box()
    maybe_1.bind(var_0)


def test_case_8():
    none_type_0 = None
    none_type_1 = None
    maybe_0 = module_0.Maybe(none_type_1, none_type_1)
    maybe_0.ap(none_type_0)


def test_case_9():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    none_type_0 = None
    var_0 = maybe_0.filter(none_type_0)
    var_1 = maybe_0.to_box()


def test_case_10():
    none_type_0 = None
    bool_0 = False
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    maybe_0.filter(maybe_0)


def test_case_11():
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.get_or_else(maybe_0)


def test_case_12():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.to_lazy()
    var_1 = maybe_0.to_either()
    var_2 = maybe_0.get_or_else(none_type_0)
    var_2.get_or_else(maybe_0)


def test_case_13():
    none_type_0 = None
    bytes_0 = b"\xe1\x14\x16\x055\xb7\n\xf2"
    maybe_0 = module_0.Maybe(bytes_0, bytes_0)
    var_0 = maybe_0.to_either()
    var_1 = var_0.ap(none_type_0)
    var_1.to_either()


def test_case_14():
    bytes_0 = b">\x1c\x99\xc9\xfb="
    bool_0 = False
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.to_either()
    var_1 = var_0.to_validation()
    var_1.get_or_else(bytes_0)


def test_case_15():
    int_0 = 1652
    int_1 = -1438
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.to_lazy()
    var_1 = var_0.bind(int_1)
    var_1.filter(int_0)


def test_case_16():
    none_type_0 = None
    bool_0 = False
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.to_lazy()
    var_1 = var_0.to_box()


def test_case_17():
    str_0 = "SY/\nR[&Xi`e<gt"
    bool_0 = True
    maybe_0 = module_0.Maybe(str_0, bool_0)
    none_type_0 = None
    var_0 = maybe_0.filter(none_type_0)
    var_1 = maybe_0.to_box()
    var_2 = maybe_0.to_try()


def test_case_18():
    int_0 = 499
    bool_0 = False
    maybe_0 = module_0.Maybe(int_0, bool_0)
    var_0 = maybe_0.to_try()
    var_0.to_lazy()


def test_case_19():
    bool_0 = True
    list_0 = [bool_0, bool_0]
    maybe_0 = module_0.Maybe(list_0, bool_0)
    var_0 = maybe_0.to_validation()
    var_1 = var_0.to_lazy()
    var_2 = var_1.to_validation()


def test_case_20():
    none_type_0 = None
    none_type_1 = None
    str_0 = "yb3mkc;66KEUn.JO^'"
    bool_0 = False
    maybe_0 = module_0.Maybe(none_type_1, str_0)
    var_0 = maybe_0.filter(none_type_0)
    var_1 = maybe_0.to_box()
    maybe_1 = module_0.Maybe(str_0, bool_0)
    var_2 = maybe_1.to_validation()
    maybe_1.filter(none_type_1)


def test_case_21():
    int_0 = -533
    maybe_0 = module_0.Maybe(int_0, int_0)
    var_0 = maybe_0.to_lazy()
    var_1 = maybe_0.to_either()
    bool_0 = True
    maybe_1 = module_0.Maybe(var_1, var_1)
    none_type_0 = None
    var_2 = maybe_0.filter(var_1)
    maybe_2 = module_0.Maybe(none_type_0, bool_0)
    var_3 = var_2.filter(var_1)
    bool_1 = var_2.__eq__(var_3)
    var_4 = maybe_1.to_box()


def test_case_22():
    bool_0 = False
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    bool_1 = maybe_0.__eq__(maybe_0)
    maybe_0.bind(maybe_0)
