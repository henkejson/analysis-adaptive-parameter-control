# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.maybe as module_0
import builtins as module_1
import typing as module_2


def test_case_0():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)


def test_case_1():
    list_0 = []
    maybe_0 = module_0.Maybe(list_0, list_0)


def test_case_2():
    bool_0 = False
    none_type_0 = None
    maybe_0 = module_0.Maybe(bool_0, none_type_0)
    bool_1 = maybe_0.__eq__(maybe_0)
    maybe_1 = module_0.Maybe(bool_1, bool_1)
    var_0 = maybe_1.map(maybe_1)
    var_1 = var_0.filter(bool_1)
    maybe_0.filter(var_0)


def test_case_3():
    bool_0 = False
    none_type_0 = None
    maybe_0 = module_0.Maybe(bool_0, none_type_0)
    maybe_1 = module_0.Maybe(bool_0, bool_0)
    maybe_2 = module_0.Maybe(bool_0, bool_0)
    none_type_1 = None
    maybe_3 = module_0.Maybe(maybe_2, none_type_1)
    var_0 = maybe_3.to_lazy()
    bool_1 = maybe_3.__eq__(maybe_3)
    var_1 = maybe_0.get_or_else(bool_1)
    maybe_4 = module_0.Maybe(var_0, var_0)
    var_2 = maybe_4.map(maybe_1)
    bool_2 = maybe_2.__eq__(var_1)
    var_3 = var_2.filter(bool_2)
    var_4 = var_3.to_validation()
    maybe_0.filter(var_2)


def test_case_4():
    bool_0 = False
    none_type_0 = None
    maybe_0 = module_0.Maybe(bool_0, none_type_0)
    var_0 = maybe_0.to_lazy()
    maybe_1 = module_0.Maybe(maybe_0, var_0)
    var_1 = maybe_1.to_lazy()
    none_type_1 = None
    bool_1 = maybe_1.__eq__(maybe_1)
    bool_2 = True
    var_2 = var_1.to_validation()
    tuple_0 = (bool_2,)
    maybe_2 = module_0.Maybe(tuple_0, bool_2)
    maybe_0.map(none_type_1)


def test_case_5():
    bytes_0 = b"jG"
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, bytes_0)
    var_0 = maybe_0.to_lazy()
    var_1 = maybe_0.map(none_type_0)
    var_2 = maybe_0.bind(var_0)
    var_3 = maybe_0.filter(var_0)
    var_4 = var_3.to_lazy()
    maybe_1 = module_0.Maybe(var_2, var_4)
    var_5 = var_1.bind(var_2)
    var_0.get_or_else(none_type_0)


def test_case_6():
    object_0 = module_1.object()
    int_0 = 0
    set_0 = {int_0, int_0, int_0}
    bool_0 = False
    maybe_0 = module_0.Maybe(set_0, bool_0)
    var_0 = maybe_0.to_lazy()
    float_0 = -2589.103
    none_type_0 = None
    maybe_1 = module_0.Maybe(float_0, none_type_0)
    maybe_1.bind(var_0)


def test_case_7():
    complex_0 = 3409.763469 - 866.769j
    bool_0 = True
    maybe_0 = module_0.Maybe(complex_0, bool_0)
    maybe_1 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.to_lazy()
    var_1 = maybe_0.to_either()
    var_2 = maybe_0.ap(var_0)
    var_3 = var_2.to_lazy()
    var_4 = maybe_0.bind(var_0)
    var_5 = var_4.to_lazy()
    var_6 = var_0.to_either()
    bool_1 = maybe_0.__eq__(bool_0)
    var_3.to_lazy()


def test_case_8():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    maybe_0.ap(none_type_0)


def test_case_9():
    list_0 = []
    int_0 = -971
    bool_0 = True
    maybe_0 = module_0.Maybe(int_0, bool_0)
    var_0 = maybe_0.get_or_else(list_0)
    var_0.to_validation()


def test_case_10():
    set_0 = set()
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.get_or_else(set_0)


def test_case_11():
    generic_0 = module_2.Generic()
    none_type_0 = None
    bool_0 = False
    maybe_0 = module_0.Maybe(generic_0, bool_0)
    var_0 = maybe_0.get_or_else(none_type_0)
    bool_1 = var_0.__eq__(generic_0)
    maybe_1 = module_0.Maybe(generic_0, generic_0)
    int_0 = 3290
    maybe_2 = module_0.Maybe(int_0, int_0)
    var_1 = maybe_2.to_either()
    var_2 = var_1.to_try()
    bool_2 = var_2.__eq__(var_1)
    var_2.to_try()


def test_case_12():
    str_0 = "%|:\\={"
    none_type_0 = None
    maybe_0 = module_0.Maybe(str_0, none_type_0)
    var_0 = maybe_0.to_either()


def test_case_13():
    none_type_0 = None
    none_type_1 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_1, bool_0)
    var_0 = maybe_0.to_box()
    var_0.ap(none_type_0)


def test_case_14():
    bool_0 = False
    none_type_0 = None
    maybe_0 = module_0.Maybe(bool_0, none_type_0)
    var_0 = maybe_0.to_box()
    bool_1 = maybe_0.__eq__(maybe_0)
    maybe_1 = module_0.Maybe(bool_1, bool_1)
    var_1 = maybe_1.map(maybe_1)
    var_2 = var_1.filter(bool_1)
    maybe_0.filter(var_1)


def test_case_15():
    bool_0 = False
    none_type_0 = None
    maybe_0 = module_0.Maybe(bool_0, none_type_0)
    var_0 = maybe_0.to_lazy()
    maybe_1 = module_0.Maybe(maybe_0, none_type_0)
    none_type_1 = None
    bool_1 = maybe_1.__eq__(maybe_1)
    tuple_0 = var_0.ap(none_type_0)
    maybe_2 = module_0.Maybe(tuple_0, bool_1)
    var_1 = maybe_2.get_or_else(none_type_1)
    var_2 = var_0.to_either()
    var_3 = maybe_2.to_lazy()
    bool_0.to_lazy()


def test_case_16():
    bool_0 = False
    none_type_0 = None
    maybe_0 = module_0.Maybe(bool_0, none_type_0)
    var_0 = maybe_0.to_lazy()
    maybe_1 = module_0.Maybe(var_0, var_0)
    var_1 = maybe_1.map(maybe_1)
    var_2 = var_1.filter(var_0)
    maybe_0.filter(var_1)


def test_case_17():
    bool_0 = False
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    none_type_0 = None
    maybe_1 = module_0.Maybe(maybe_0, none_type_0)
    var_0 = maybe_1.to_lazy()
    bool_1 = maybe_1.__eq__(maybe_1)
    bool_2 = True
    var_1 = var_0.to_validation()
    tuple_0 = (bool_2,)
    maybe_2 = module_0.Maybe(tuple_0, bool_2)
    var_2 = var_0.bind(tuple_0)
    var_3 = maybe_2.to_try()


def test_case_18():
    bool_0 = False
    none_type_0 = None
    maybe_0 = module_0.Maybe(bool_0, none_type_0)
    maybe_1 = module_0.Maybe(bool_0, bool_0)
    none_type_1 = None
    maybe_2 = module_0.Maybe(maybe_0, none_type_1)
    var_0 = maybe_2.to_lazy()
    var_1 = var_0.to_validation()
    var_2 = maybe_2.to_lazy()
    tuple_0 = (maybe_1,)
    var_3 = maybe_0.get_or_else(var_1)
    maybe_3 = module_0.Maybe(tuple_0, var_0)
    var_4 = maybe_0.to_try()
    var_5 = maybe_3.map(maybe_1)
    bool_1 = var_0.__eq__(var_1)
    var_6 = var_5.filter(bool_1)
    maybe_0.filter(var_5)


def test_case_19():
    bool_0 = False
    none_type_0 = None
    maybe_0 = module_0.Maybe(bool_0, none_type_0)
    var_0 = maybe_0.to_box()
    maybe_1 = module_0.Maybe(var_0, var_0)
    bool_1 = maybe_1.__eq__(maybe_1)
    maybe_2 = module_0.Maybe(bool_1, bool_1)
    var_1 = maybe_2.map(maybe_2)
    var_2 = var_1.filter(bool_1)
    var_3 = var_2.to_validation()
    maybe_0.filter(var_1)


def test_case_20():
    list_0 = []
    object_0 = module_1.object(*list_0)
    list_1 = [object_0]
    tuple_0 = (list_1,)
    bool_0 = False
    maybe_0 = module_0.Maybe(tuple_0, bool_0)
    var_0 = maybe_0.to_validation()


def test_case_21():
    bool_0 = False
    none_type_0 = None
    maybe_0 = module_0.Maybe(bool_0, none_type_0)
    var_0 = maybe_0.to_lazy()
    maybe_1 = module_0.Maybe(maybe_0, none_type_0)
    none_type_1 = None
    bool_1 = maybe_1.__eq__(maybe_1)
    tuple_0 = (var_0,)
    maybe_2 = module_0.Maybe(tuple_0, bool_1)
    var_1 = maybe_2.get_or_else(none_type_1)
    var_2 = var_0.to_either()
    var_0.get_or_else(var_0)


def test_case_22():
    bool_0 = False
    none_type_0 = None
    maybe_0 = module_0.Maybe(bool_0, none_type_0)
    var_0 = maybe_0.to_lazy()
    maybe_1 = module_0.Maybe(bool_0, bool_0)
    none_type_1 = None
    maybe_2 = module_0.Maybe(maybe_1, none_type_1)
    var_1 = maybe_2.to_lazy()
    bool_1 = maybe_2.__eq__(maybe_2)
    var_2 = var_1.to_validation()
    bool_2 = maybe_2.__eq__(maybe_1)
    var_3 = maybe_0.to_lazy()
    var_2.to_validation()
