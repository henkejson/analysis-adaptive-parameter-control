# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.maybe as module_0


def test_case_0():
    int_0 = 430
    maybe_0 = module_0.Maybe(int_0, int_0)


def test_case_1():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)


def test_case_2():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.map(maybe_0)
    var_1 = maybe_0.ap(maybe_0)
    bool_1 = var_0.__eq__(var_1)
    var_2 = maybe_0.to_lazy()
    bool_2 = True
    bool_2.to_try()


def test_case_3():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    bool_0 = maybe_0.__eq__(none_type_0)
    var_0 = maybe_0.to_try()
    var_1 = maybe_0.to_lazy()


def test_case_4():
    tuple_0 = ()
    bool_0 = False
    none_type_0 = None
    maybe_0 = module_0.Maybe(bool_0, none_type_0)
    maybe_0.map(tuple_0)


def test_case_5():
    bool_0 = False
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    bool_1 = True
    maybe_1 = module_0.Maybe(maybe_0, bool_1)
    var_0 = maybe_0.to_either()
    var_1 = maybe_1.bind(var_0)
    var_2 = maybe_1.to_try()
    var_2.to_either()


def test_case_6():
    int_0 = 867
    none_type_0 = None
    none_type_1 = None
    maybe_0 = module_0.Maybe(int_0, none_type_1)
    str_0 = "\n    The Try control gives us the ability write safe code\n    without focusing on try-catch blocks in the presence of exceptions.\n    "
    var_0 = maybe_0.get_or_else(str_0)
    var_1 = maybe_0.to_lazy()
    var_2 = maybe_0.to_lazy()
    bool_0 = maybe_0.__eq__(none_type_0)
    var_3 = maybe_0.to_validation()
    maybe_0.bind(none_type_1)


def test_case_7():
    dict_0 = {}
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.to_lazy()
    var_1 = var_0.to_box()
    bool_0 = True
    tuple_0 = (bool_0, dict_0)
    maybe_1 = module_0.Maybe(dict_0, none_type_0)
    maybe_1.ap(tuple_0)


def test_case_8():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.ap(maybe_0)
    var_1 = maybe_0.filter(var_0)
    bool_1 = var_1.__eq__(var_0)
    list_0 = [var_0, var_1]
    var_2 = var_1.ap(var_0)
    var_3 = var_0.get_or_else(list_0)
    var_4 = var_1.to_validation()


def test_case_9():
    none_type_0 = None
    bool_0 = False
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    maybe_0.filter(none_type_0)


def test_case_10():
    int_0 = 920
    bool_0 = True
    maybe_0 = module_0.Maybe(int_0, bool_0)
    var_0 = maybe_0.to_either()
    var_0.to_either()


def test_case_11():
    bool_0 = False
    none_type_0 = None
    maybe_0 = module_0.Maybe(bool_0, none_type_0)
    var_0 = maybe_0.to_either()
    var_0.filter(none_type_0)


def test_case_12():
    float_0 = 3408.546
    maybe_0 = module_0.Maybe(float_0, float_0)
    var_0 = maybe_0.to_box()
    var_1 = var_0.to_either()
    var_2 = var_1.to_lazy()
    var_3 = var_2.to_validation()


def test_case_13():
    bool_0 = False
    bool_1 = False
    maybe_0 = module_0.Maybe(bool_0, bool_1)
    var_0 = maybe_0.to_box()
    var_0.to_box()


def test_case_14():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.map(maybe_0)
    var_1 = maybe_0.ap(maybe_0)
    bool_1 = var_0.__eq__(var_1)
    var_2 = maybe_0.to_lazy()
    bool_2 = True
    maybe_1 = module_0.Maybe(bool_2, bool_2)
    var_3 = maybe_1.to_try()
    var_4 = maybe_0.to_validation()
    var_5 = maybe_1.to_validation()


def test_case_15():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    bool_0 = maybe_0.__eq__(none_type_0)
    var_0 = maybe_0.to_try()
    var_1 = maybe_0.to_lazy()
    var_2 = maybe_0.to_validation()
    var_0.to_lazy()


def test_case_16():
    int_0 = 867
    none_type_0 = None
    none_type_1 = None
    maybe_0 = module_0.Maybe(int_0, none_type_1)
    str_0 = "\n    The Try control gives us the ability write safe code\n    without focusing on try-catch blocks in the presence of exceptions.\n    "
    var_0 = maybe_0.get_or_else(str_0)
    var_1 = maybe_0.to_lazy()
    var_2 = maybe_0.to_lazy()
    bool_0 = maybe_0.__eq__(none_type_0)
    var_3 = maybe_0.to_validation()
    int_1 = 1
    var_4 = var_2.to_validation()
    var_5 = var_4.to_either()
    maybe_1 = module_0.Maybe(int_0, int_1)
    bool_1 = True
    maybe_2 = module_0.Maybe(maybe_1, bool_1)
    int_2 = 935
    var_2.get_or_else(int_2)


def test_case_17():
    int_0 = 1106
    maybe_0 = module_0.Maybe(int_0, int_0)
    var_0 = maybe_0.to_either()
    var_1 = var_0.to_try()
    var_2 = maybe_0.filter(int_0)
    var_3 = var_2.to_validation()
    var_4 = maybe_0.to_try()
    var_5 = var_3.to_either()
    bool_0 = var_2.__eq__(var_0)
    var_6 = var_2.filter(var_3)
    var_7 = maybe_0.to_either()
    str_0 = ".SB#(XNMb.]"
    var_8 = var_2.map(str_0)
    bool_1 = maybe_0.__eq__(var_2)
    bool_2 = True
    maybe_1 = module_0.Maybe(maybe_0, bool_2)
    var_9 = maybe_1.map(var_7)
    var_10 = maybe_1.ap(bool_2)
    var_11 = maybe_0.to_validation()
    var_1.to_try()


def test_case_18():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    none_type_0 = None
    var_0 = maybe_0.ap(none_type_0)
    var_1 = var_0.to_lazy()
    var_2 = maybe_0.bind(maybe_0)
    var_3 = maybe_0.filter(maybe_0)
    var_4 = var_1.map(var_0)
    maybe_1 = module_0.Maybe(var_4, none_type_0)
    var_5 = var_1.to_try()
    maybe_1.filter(var_4)


def test_case_19():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.to_lazy()
    var_1 = maybe_0.to_either()
    var_2 = maybe_0.to_validation()
    var_3 = var_1.to_box()
    var_4 = maybe_0.to_lazy()
    none_type_1 = None
    maybe_1 = module_0.Maybe(none_type_1, none_type_1)
    maybe_2 = module_0.Maybe(var_3, var_2)
    var_5 = maybe_1.to_validation()
    var_6 = var_5.to_either()
    var_7 = var_3.to_validation()
    var_8 = maybe_1.to_lazy()
    bool_0 = maybe_1.__eq__(maybe_1)
    var_9 = maybe_2.to_lazy()
    var_10 = var_9.bind(none_type_0)
    var_11 = var_8.to_try()
    var_12 = maybe_1.to_try()
    var_13 = var_8.map(var_5)
    var_12.to_try()
