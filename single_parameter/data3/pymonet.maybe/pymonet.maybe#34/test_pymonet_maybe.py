# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.maybe as module_0
import typing as module_1
import builtins as module_2


def test_case_0():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)


def test_case_1():
    bool_0 = False
    maybe_0 = module_0.Maybe(bool_0, bool_0)


def test_case_2():
    int_0 = -1252
    maybe_0 = module_0.Maybe(int_0, int_0)
    bool_0 = maybe_0.__eq__(maybe_0)
    bool_1 = maybe_0.__eq__(int_0)
    var_0 = maybe_0.to_validation()
    bool_1.to_either()


def test_case_3():
    bool_0 = True
    str_0 = "\n        Take function, store it and call with Task value during calling fork function.\n        Return result of called.\n\n        :param fn: mapper function\n        :type fn: Function(value) -> Task[reject, mapped_value]\n        :returns:  new Task with mapper resolve attribute\n        :rtype: Task[reject, mapped_value]\n        "
    bool_1 = True
    maybe_0 = module_0.Maybe(str_0, bool_1)
    var_0 = maybe_0.map(bool_0)


def test_case_4():
    bool_0 = True
    generic_0 = module_1.Generic()
    none_type_0 = None
    maybe_0 = module_0.Maybe(generic_0, none_type_0)
    maybe_0.map(bool_0)


def test_case_5():
    none_type_0 = None
    none_type_1 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.bind(none_type_1)
    var_1 = var_0.to_lazy()
    var_1.get_or_else(none_type_0)


def test_case_6():
    bool_0 = True
    bool_1 = True
    none_type_0 = None
    maybe_0 = module_0.Maybe(bool_1, none_type_0)
    maybe_0.bind(bool_0)


def test_case_7():
    int_0 = 2479
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.ap(int_0)


def test_case_8():
    none_type_0 = None
    none_type_1 = None
    maybe_0 = module_0.Maybe(none_type_1, none_type_1)
    maybe_0.ap(none_type_0)


def test_case_9():
    str_0 = "\n        Take function, store it and call with Task value during calling fork function.\n        Return result of called.\n\n        :param fn: mapper function\n        :type fn: Function(value) -> Task[reject, mapped_value]\n        :returns:  new Task with mapper resolve attribute\n        :rtype: Task[reject, mapped_value]\n        "
    bool_0 = True
    maybe_0 = module_0.Maybe(str_0, bool_0)
    var_0 = maybe_0.to_try()
    var_1 = maybe_0.filter(maybe_0)
    var_0.to_box()


def test_case_10():
    int_0 = 0
    maybe_0 = module_0.Maybe(int_0, int_0)
    none_type_0 = None
    maybe_0.filter(none_type_0)


def test_case_11():
    object_0 = module_2.object()
    bool_0 = False
    maybe_0 = module_0.Maybe(object_0, bool_0)
    bool_1 = True
    maybe_1 = module_0.Maybe(object_0, bool_1)
    var_0 = maybe_1.get_or_else(maybe_1)
    maybe_0.filter(var_0)


def test_case_12():
    none_type_0 = None
    bool_0 = False
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.get_or_else(maybe_0)


def test_case_13():
    none_type_0 = None
    none_type_1 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_1, bool_0)
    var_0 = maybe_0.to_either()
    var_0.filter(none_type_0)


def test_case_14():
    object_0 = module_2.object()
    bool_0 = True
    maybe_0 = module_0.Maybe(object_0, bool_0)
    bool_1 = False
    maybe_1 = module_0.Maybe(object_0, bool_1)
    var_0 = maybe_0.filter(bool_0)
    var_1 = maybe_1.to_either()
    maybe_1.filter(var_1)


def test_case_15():
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.map(maybe_0)
    var_1 = var_0.map(none_type_0)
    var_2 = var_1.to_lazy()
    var_3 = var_1.map(var_2)
    var_4 = var_3.to_either()
    var_5 = var_4.to_try()
    var_6 = var_0.filter(var_4)
    var_7 = var_3.to_box()


def test_case_16():
    bool_0 = True
    none_type_0 = None
    maybe_0 = module_0.Maybe(bool_0, none_type_0)
    var_0 = maybe_0.to_box()
    var_1 = var_0.to_lazy()
    bool_1 = False
    var_2 = var_0.to_try()
    none_type_1 = None
    maybe_1 = module_0.Maybe(bool_1, none_type_1)
    var_3 = maybe_1.to_lazy()
    var_4 = maybe_0.to_box()
    var_5 = maybe_1.to_lazy()
    var_6 = var_0.to_validation()
    var_3.get_or_else(bool_0)


def test_case_17():
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.to_lazy()
    bool_1 = var_0.__eq__(var_0)
    var_1 = var_0.to_either()
    var_2 = var_1.to_validation()


def test_case_18():
    str_0 = "\n        Take function and applied this function with monad value and returns new monad with mapped value.\n\n        :params mapper: function to apply on monad value\n        :type mapper: Function(A) -> B\n        :returns: for successfully new Try with mapped value, othercase copy of self\n        :rtype: Try[B]\n        "
    bool_0 = False
    none_type_0 = None
    maybe_0 = module_0.Maybe(bool_0, none_type_0)
    maybe_1 = module_0.Maybe(str_0, bool_0)
    var_0 = maybe_1.to_lazy()
    var_1 = var_0.ap(maybe_0)
    var_2 = var_0.to_try()


def test_case_19():
    int_0 = 0
    maybe_0 = module_0.Maybe(int_0, int_0)
    var_0 = maybe_0.to_try()
    var_0.to_validation()


def test_case_20():
    bool_0 = False
    float_0 = -3439.643146
    bool_1 = True
    maybe_0 = module_0.Maybe(float_0, bool_1)
    var_0 = maybe_0.to_validation()
    var_1 = var_0.to_box()
    var_2 = var_1.to_lazy()
    var_3 = var_2.ap(bool_0)
    none_type_0 = None
    none_type_1 = None
    maybe_1 = module_0.Maybe(none_type_1, none_type_1)
    var_4 = maybe_1.to_validation()
    var_5 = var_4.to_either()
    var_5.ap(none_type_0)


def test_case_21():
    bool_0 = False
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.to_either()
    var_1 = maybe_0.to_either()
    maybe_1 = module_0.Maybe(var_0, maybe_0)
    var_2 = maybe_1.ap(bool_0)
    var_3 = maybe_0.to_either()
    bool_1 = maybe_0.__eq__(maybe_0)
    var_3.bind(var_3)


def test_case_22():
    int_0 = -1252
    maybe_0 = module_0.Maybe(int_0, int_0)
    maybe_1 = module_0.Maybe(maybe_0, maybe_0)
    var_0 = maybe_0.to_lazy()
    bool_0 = maybe_0.__eq__(maybe_1)
    var_1 = maybe_1.to_validation()
    var_2 = maybe_0.get_or_else(maybe_1)
    var_3 = maybe_0.filter(var_1)
    var_4 = var_2.ap(bool_0)
    maybe_2 = module_0.Maybe(var_4, var_0)
    var_5 = var_2.get_or_else(var_2)
    var_6 = var_4.filter(var_2)
    var_7 = var_3.get_or_else(var_4)
    var_8 = var_7.map(var_1)
    var_9 = var_4.to_try()
    var_10 = maybe_1.filter(maybe_2)
    maybe_3 = module_0.Maybe(var_9, maybe_2)
    var_11 = maybe_2.to_try()
    var_12 = var_4.ap(var_10)
    var_13 = var_6.filter(var_4)
