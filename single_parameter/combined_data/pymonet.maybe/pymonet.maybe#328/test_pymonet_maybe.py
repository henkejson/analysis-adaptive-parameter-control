# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.maybe as module_0
import builtins as module_1


def test_case_0():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)


def test_case_1():
    bool_0 = False
    maybe_0 = module_0.Maybe(bool_0, bool_0)


def test_case_2():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.to_either()
    var_1 = maybe_0.bind(maybe_0)
    bool_1 = maybe_0.__eq__(bool_0)
    var_2 = maybe_0.get_or_else(maybe_0)
    var_3 = var_2.to_lazy()
    bool_2 = var_2.__eq__(var_3)
    var_4 = maybe_0.bind(var_2)
    var_5 = maybe_0.ap(bool_0)


def test_case_3():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.to_try()
    var_1 = maybe_0.to_lazy()
    var_2 = maybe_0.map(var_1)
    var_3 = maybe_0.filter(var_0)


def test_case_4():
    bool_0 = False
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.to_lazy()
    var_1 = maybe_0.to_either()
    maybe_0.map(maybe_0)


def test_case_5():
    list_0 = []
    bool_0 = True
    maybe_0 = module_0.Maybe(list_0, bool_0)
    maybe_1 = module_0.Maybe(list_0, list_0)
    var_0 = maybe_1.to_try()
    maybe_1.bind(var_0)


def test_case_6():
    none_type_0 = None
    none_type_1 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.ap(none_type_1)
    var_1 = var_0.to_either()
    var_2 = var_1.to_try()
    var_3 = var_2.get_or_else(none_type_0)


def test_case_7():
    str_0 = "4N>l?Esp["
    bool_0 = False
    maybe_0 = module_0.Maybe(str_0, bool_0)
    maybe_0.ap(str_0)


def test_case_8():
    int_0 = 3391
    list_0 = [int_0, int_0, int_0]
    dict_0 = {int_0: list_0}
    maybe_0 = module_0.Maybe(dict_0, list_0)
    bool_0 = maybe_0.__eq__(int_0)
    var_0 = maybe_0.to_box()
    bool_1 = True
    bool_2 = False
    maybe_1 = module_0.Maybe(maybe_0, bool_2)
    maybe_2 = module_0.Maybe(dict_0, maybe_1)
    var_1 = maybe_1.get_or_else(int_0)
    var_2 = var_0.to_validation()
    var_3 = var_1.bind(var_1)
    var_4 = maybe_2.to_either()
    var_2.filter(bool_1)


def test_case_9():
    bool_0 = False
    set_0 = {bool_0, bool_0, bool_0, bool_0}
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.to_either()
    var_0.ap(set_0)


def test_case_10():
    float_0 = -1434.63291
    bool_0 = True
    maybe_0 = module_0.Maybe(float_0, bool_0)
    var_0 = maybe_0.to_box()
    var_1 = var_0.to_either()
    bool_1 = False
    maybe_1 = module_0.Maybe(float_0, bool_1)


def test_case_11():
    object_0 = module_1.object()
    bool_0 = False
    maybe_0 = module_0.Maybe(object_0, bool_0)
    var_0 = maybe_0.to_box()
    var_1 = maybe_0.to_try()
    var_0.filter(var_1)


def test_case_12():
    bool_0 = False
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.to_validation()
    dict_0 = {}
    maybe_1 = module_0.Maybe(dict_0, dict_0)
    var_1 = maybe_1.to_lazy()
    var_1.filter(var_1)


def test_case_13():
    bytes_0 = b"\x0cN\xbe\xd0C\x15\xb1kTqg"
    maybe_0 = module_0.Maybe(bytes_0, bytes_0)
    var_0 = maybe_0.to_validation()
    var_1 = var_0.to_box()
    maybe_1 = module_0.Maybe(var_1, var_0)
    var_2 = maybe_1.to_either()
    maybe_2 = module_0.Maybe(var_1, maybe_0)
    var_3 = var_2.to_validation()


def test_case_14():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.to_either()
    var_1 = maybe_0.to_either()
    var_2 = maybe_0.bind(maybe_0)
    bool_1 = maybe_0.__eq__(bool_0)
    var_3 = maybe_0.get_or_else(maybe_0)
    var_4 = var_3.to_lazy()
    bool_2 = var_3.__eq__(var_4)
    var_5 = var_2.to_box()
    var_6 = var_4.to_try()
    bool_3 = var_4.__eq__(maybe_0)
    var_1.get_or_else(bool_3)


def test_case_15():
    int_0 = 3403
    list_0 = [int_0, int_0, int_0]
    dict_0 = {int_0: list_0}
    maybe_0 = module_0.Maybe(dict_0, list_0)
    var_0 = maybe_0.bind(int_0)
    var_1 = maybe_0.bind(var_0)
    bool_0 = var_0.__eq__(int_0)
    bool_1 = False
    maybe_1 = module_0.Maybe(bool_1, bool_1)
    var_2 = var_0.get_or_else(var_0)
    var_3 = maybe_1.to_lazy()
    var_4 = var_2.map(var_2)
    var_5 = var_1.to_either()
    var_6 = maybe_0.to_either()
    var_7 = var_0.filter(var_6)
    var_8 = var_3.to_box()


def test_case_16():
    int_0 = 3403
    list_0 = [int_0, int_0, int_0]
    dict_0 = {int_0: list_0}
    maybe_0 = module_0.Maybe(dict_0, list_0)
    bool_0 = maybe_0.__eq__(int_0)
    maybe_1 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.get_or_else(maybe_0)
    bool_1 = var_0.__eq__(var_0)
    var_1 = var_0.filter(var_0)
    maybe_1.filter(bool_1)


def test_case_17():
    int_0 = 3403
    list_0 = [int_0, int_0, int_0]
    dict_0 = {int_0: list_0}
    maybe_0 = module_0.Maybe(dict_0, list_0)
    bool_0 = maybe_0.__eq__(int_0)
    maybe_1 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.get_or_else(maybe_0)
    var_1 = var_0.filter(var_0)
    maybe_1.filter(bool_0)


def test_case_18():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.to_box()
    bool_0 = maybe_0.__eq__(maybe_0)
    var_0.get_or_else(none_type_0)


def test_case_19():
    int_0 = 3403
    list_0 = [int_0, int_0, int_0]
    dict_0 = {int_0: list_0}
    maybe_0 = module_0.Maybe(dict_0, list_0)
    none_type_0 = None
    var_0 = maybe_0.bind(none_type_0)
    bool_0 = maybe_0.__eq__(list_0)
    var_1 = maybe_0.get_or_else(none_type_0)
    var_2 = var_0.to_lazy()
    var_3 = var_0.filter(var_1)
    bool_1 = var_3.__eq__(maybe_0)
    var_4 = var_2.to_either()
    var_1.bind(maybe_0)
