# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.maybe as module_0
import builtins as module_1
import typing as module_2


def test_case_0():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)


def test_case_1():
    set_0 = set()
    maybe_0 = module_0.Maybe(set_0, set_0)


def test_case_2():
    object_0 = module_1.object()
    maybe_0 = module_0.Maybe(object_0, object_0)
    complex_0 = 1449.421 + 72.1j
    int_0 = 1719
    maybe_1 = module_0.Maybe(int_0, int_0)
    var_0 = maybe_1.to_try()
    str_0 = "Sum[value={}]"
    maybe_2 = module_0.Maybe(str_0, str_0)
    var_1 = maybe_2.map(maybe_2)
    var_2 = maybe_0.bind(complex_0)
    bool_0 = maybe_2.__eq__(maybe_2)


def test_case_3():
    int_0 = -524
    str_0 = "Sum[value={}]"
    maybe_0 = module_0.Maybe(str_0, str_0)
    tuple_0 = (maybe_0,)
    maybe_1 = module_0.Maybe(int_0, tuple_0)
    bool_0 = maybe_1.__eq__(str_0)
    maybe_2 = module_0.Maybe(str_0, str_0)
    var_0 = maybe_2.get_or_else(int_0)
    var_0.to_try()


def test_case_4():
    bool_0 = True
    none_type_0 = None
    list_0 = [none_type_0, none_type_0, none_type_0]
    maybe_0 = module_0.Maybe(none_type_0, list_0)
    bool_1 = maybe_0.__eq__(bool_0)
    none_type_1 = None
    var_0 = maybe_0.ap(list_0)
    var_1 = maybe_0.filter(var_0)
    var_2 = maybe_0.bind(var_0)
    maybe_1 = module_0.Maybe(none_type_1, none_type_1)
    maybe_2 = module_0.Maybe(none_type_1, maybe_1)
    var_3 = maybe_0.filter(var_0)
    var_4 = maybe_2.to_lazy()
    var_5 = var_2.bind(maybe_2)
    maybe_1.ap(maybe_1)


def test_case_5():
    bool_0 = True
    set_0 = set()
    maybe_0 = module_0.Maybe(set_0, set_0)
    maybe_0.bind(bool_0)


def test_case_6():
    bool_0 = True
    none_type_0 = None
    maybe_0 = module_0.Maybe(bool_0, none_type_0)
    maybe_1 = module_0.Maybe(bool_0, maybe_0)
    var_0 = maybe_1.ap(bool_0)
    var_1 = maybe_1.filter(maybe_0)
    bool_1 = var_1.to_box()
    maybe_0.filter(maybe_0)


def test_case_7():
    none_type_0 = None
    bool_0 = False
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    maybe_0.ap(none_type_0)


def test_case_8():
    bool_0 = False
    none_type_0 = None
    maybe_0 = module_0.Maybe(bool_0, none_type_0)
    maybe_1 = module_0.Maybe(bool_0, maybe_0)
    var_0 = maybe_1.to_lazy()
    var_1 = maybe_1.filter(maybe_0)
    maybe_0.filter(maybe_0)


def test_case_9():
    int_0 = 1
    bool_0 = True
    maybe_0 = module_0.Maybe(int_0, bool_0)
    var_0 = maybe_0.to_validation()
    maybe_1 = module_0.Maybe(int_0, int_0)
    var_1 = maybe_1.to_box()
    var_2 = maybe_1.get_or_else(int_0)
    maybe_2 = module_0.Maybe(var_1, var_1)
    var_1.bind(var_1)


def test_case_10():
    bool_0 = False
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.get_or_else(maybe_0)
    bool_1 = maybe_0.__eq__(bool_0)
    maybe_0.filter(var_0)


def test_case_11():
    object_0 = module_1.object()
    int_0 = 1719
    maybe_0 = module_0.Maybe(int_0, int_0)
    var_0 = maybe_0.to_either()
    var_1 = maybe_0.ap(var_0)
    var_2 = var_1.filter(maybe_0)
    var_3 = var_1.map(var_1)
    bool_0 = var_3.__eq__(var_3)
    var_4 = var_3.to_box()
    var_5 = var_1.bind(bool_0)
    var_6 = maybe_0.to_lazy()


def test_case_12():
    generic_0 = module_2.Generic()
    none_type_0 = None
    maybe_0 = module_0.Maybe(generic_0, none_type_0)
    var_0 = maybe_0.to_either()
    bool_0 = True
    int_0 = 3579
    dict_0 = {int_0: int_0, int_0: int_0}
    bool_1 = True
    maybe_1 = module_0.Maybe(dict_0, bool_1)
    var_1 = maybe_1.ap(bool_0)
    var_2 = var_1.to_try()
    var_3 = var_2.get_or_else(var_0)


def test_case_13():
    bool_0 = False
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    maybe_1 = module_0.Maybe(bool_0, maybe_0)
    var_0 = maybe_0.to_box()
    var_1 = maybe_1.filter(maybe_0)
    bool_1 = none_type_0.__eq__(none_type_0)
    maybe_0.filter(maybe_0)


def test_case_14():
    set_0 = set()
    maybe_0 = module_0.Maybe(set_0, set_0)
    var_0 = maybe_0.to_lazy()


def test_case_15():
    int_0 = -117
    str_0 = "\n        Return monad value when is successfully.\n        Othercase return default_value argument.\n\n        :params default_value: value to return when monad is not successfully.\n        :type default_value: B\n        :returns: monad value\n        :rtype: A | B\n        "
    maybe_0 = module_0.Maybe(str_0, str_0)
    var_0 = maybe_0.to_try()
    var_0.ap(int_0)


def test_case_16():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.to_try()
    var_0.to_try()


def test_case_17():
    bool_0 = False
    tuple_0 = (bool_0, bool_0, bool_0)
    maybe_0 = module_0.Maybe(tuple_0, tuple_0)
    var_0 = maybe_0.to_validation()


def test_case_18():
    tuple_0 = ()
    maybe_0 = module_0.Maybe(tuple_0, tuple_0)
    var_0 = maybe_0.to_validation()
    var_0.get_or_else(tuple_0)


def test_case_19():
    none_type_0 = None
    bool_0 = False
    bytes_0 = b"N\x8d\xc6`\x84\xf8\xba\xb5Y\xb7kl"
    maybe_0 = module_0.Maybe(bytes_0, bytes_0)
    var_0 = maybe_0.to_lazy()
    int_0 = 2385
    bool_1 = False
    maybe_1 = module_0.Maybe(int_0, bool_1)
    var_1 = maybe_1.to_try()
    bool_2 = True
    maybe_2 = module_0.Maybe(var_1, bool_2)
    var_2 = maybe_2.ap(var_0)
    var_3 = var_2.ap(bool_0)
    var_4 = maybe_0.map(none_type_0)
    bool_3 = var_2.__eq__(maybe_0)
    var_5 = var_0.to_box()
    var_6 = var_0.to_box()


def test_case_20():
    int_0 = -1819
    none_type_0 = None
    bool_0 = False
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.to_lazy()
    var_1 = var_0.bind(int_0)
    var_1.to_box()


def test_case_21():
    bool_0 = False
    bytes_0 = b"N\x8d\xc6`\x84\xf8\xba\xb5Y\xb7kl"
    maybe_0 = module_0.Maybe(bytes_0, bytes_0)
    int_0 = 2385
    bool_1 = False
    maybe_1 = module_0.Maybe(int_0, bool_1)
    var_0 = maybe_1.to_try()
    bool_2 = True
    maybe_2 = module_0.Maybe(var_0, bool_2)
    var_1 = maybe_2.ap(bool_0)
    var_2 = var_1.ap(bool_0)
    bool_3 = var_1.__eq__(maybe_0)
    var_3 = maybe_0.to_box()


def test_case_22():
    object_0 = module_1.object()
    maybe_0 = module_0.Maybe(object_0, object_0)
    int_0 = 1719
    maybe_1 = module_0.Maybe(int_0, int_0)
    var_0 = maybe_0.to_validation()
    var_1 = maybe_1.bind(int_0)
    bool_0 = False
    maybe_2 = module_0.Maybe(int_0, bool_0)
    var_2 = maybe_1.ap(bool_0)
    var_3 = maybe_2.to_lazy()
    var_4 = var_3.to_try()
    var_5 = var_2.get_or_else(bool_0)
    bool_1 = maybe_2.__eq__(maybe_2)
    var_6 = maybe_1.get_or_else(bool_0)
    var_6.map(var_0)
