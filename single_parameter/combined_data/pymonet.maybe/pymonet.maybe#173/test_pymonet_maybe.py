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
    none_type_0 = None
    bool_0 = False
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    bool_1 = maybe_0.__eq__(maybe_0)
    bool_2 = False
    maybe_1 = module_0.Maybe(bool_0, bool_2)
    var_0 = maybe_0.to_box()


def test_case_3():
    int_0 = -5699
    int_1 = 2375
    bool_0 = True
    maybe_0 = module_0.Maybe(int_1, bool_0)
    bool_1 = maybe_0.__eq__(int_0)


def test_case_4():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    none_type_0 = None
    var_0 = maybe_0.map(none_type_0)
    var_1 = maybe_0.ap(none_type_0)
    var_2 = var_0.to_lazy()
    var_3 = maybe_0.map(bool_0)
    var_4 = maybe_0.ap(var_0)
    var_5 = var_4.bind(var_1)
    bool_1 = maybe_0.__eq__(bool_0)


def test_case_5():
    float_0 = -1327.375923600444
    none_type_0 = None
    bool_0 = False
    maybe_0 = module_0.Maybe(float_0, bool_0)
    var_0 = maybe_0.to_either()
    bool_1 = True
    maybe_1 = module_0.Maybe(float_0, bool_1)
    var_1 = maybe_1.map(float_0)
    var_2 = var_1.bind(var_0)
    var_3 = var_2.filter(none_type_0)
    var_4 = var_3.filter(var_0)
    var_5 = var_4.ap(float_0)
    dict_0 = {float_0: float_0, float_0: float_0}
    maybe_2 = module_0.Maybe(dict_0, dict_0)
    var_6 = maybe_2.to_either()
    var_7 = maybe_2.to_lazy()
    maybe_3 = module_0.Maybe(var_3, float_0)
    maybe_0.map(var_1)


def test_case_6():
    int_0 = -1163
    maybe_0 = module_0.Maybe(int_0, int_0)
    var_0 = maybe_0.to_lazy()
    maybe_1 = module_0.Maybe(int_0, int_0)
    var_1 = maybe_1.bind(maybe_0)


def test_case_7():
    bool_0 = False
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.to_either()
    maybe_0.bind(var_0)


def test_case_8():
    float_0 = -2416.28559
    maybe_0 = module_0.Maybe(float_0, float_0)
    var_0 = maybe_0.to_box()
    var_1 = maybe_0.bind(var_0)
    maybe_1 = module_0.Maybe(maybe_0, float_0)
    var_2 = maybe_1.ap(float_0)
    var_3 = maybe_1.bind(maybe_1)
    bool_0 = var_2.__eq__(var_2)


def test_case_9():
    list_0 = []
    maybe_0 = module_0.Maybe(list_0, list_0)
    bool_0 = True
    maybe_1 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.to_validation()
    var_1 = maybe_0.ap(maybe_1)
    var_2 = maybe_1.to_validation()
    var_3 = var_0.to_box()
    maybe_0.filter(list_0)


def test_case_10():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    none_type_0 = None
    var_0 = maybe_0.map(none_type_0)
    var_1 = maybe_0.ap(none_type_0)
    var_2 = var_1.filter(var_1)
    var_3 = maybe_0.bind(none_type_0)
    var_4 = module_1.Generic()
    var_5 = var_3.bind(var_3)
    bool_1 = var_1.__eq__(bool_0)


def test_case_11():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    maybe_0.filter(none_type_0)


def test_case_12():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    none_type_0 = None
    var_0 = maybe_0.map(none_type_0)
    var_1 = maybe_0.ap(none_type_0)
    var_2 = var_0.to_lazy()
    var_3 = maybe_0.map(bool_0)
    maybe_1 = module_0.Maybe(var_1, var_3)
    var_4 = maybe_1.ap(var_1)
    var_5 = maybe_1.get_or_else(var_4)
    var_6 = var_0.get_or_else(var_5)
    var_7 = var_0.filter(var_1)


def test_case_13():
    str_0 = "\n        Transform Box into Right either.\n\n        :returns: right Either monad with previous value\n        :rtype: Right[A]\n        "
    tuple_0 = ()
    bool_0 = False
    maybe_0 = module_0.Maybe(tuple_0, bool_0)
    var_0 = maybe_0.get_or_else(str_0)


def test_case_14():
    none_type_0 = None
    bool_0 = True
    bool_1 = True
    maybe_0 = module_0.Maybe(bool_0, bool_1)
    var_0 = maybe_0.bind(none_type_0)
    var_1 = var_0.to_either()
    var_2 = var_1.to_lazy()


def test_case_15():
    dict_0 = {}
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.to_either()
    maybe_1 = module_0.Maybe(dict_0, dict_0)
    var_1 = maybe_1.to_validation()
    tuple_0 = ()
    maybe_2 = module_0.Maybe(tuple_0, tuple_0)


def test_case_16():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.to_lazy()


def test_case_17():
    float_0 = -3542.40457
    maybe_0 = module_0.Maybe(float_0, float_0)
    var_0 = maybe_0.to_try()
    var_1 = maybe_0.to_validation()
    var_2 = maybe_0.to_either()
    var_3 = var_2.ap(float_0)
    var_4 = maybe_0.get_or_else(var_3)
    var_5 = maybe_0.filter(var_2)
    var_6 = var_5.to_lazy()
    var_7 = module_2.object()
    var_2.get_or_else(float_0)


def test_case_18():
    bytes_0 = b",t\x99\xddG\x83\xd0\xbbxf\xd7\x89\x9e\xc6\xb3"
    bool_0 = False
    bool_1 = False
    maybe_0 = module_0.Maybe(bool_0, bool_1)
    bool_2 = maybe_0.__eq__(bytes_0)
    maybe_1 = maybe_0.to_try()
    bool_3 = maybe_0.to_try()


def test_case_19():
    str_0 = '"'
    maybe_0 = module_0.Maybe(str_0, str_0)
    var_0 = maybe_0.to_validation()
    var_1 = var_0.to_try()
    var_1.to_validation()


def test_case_20():
    bytes_0 = b"n\xeaN\x0f\x8f\xb1\x00\xd1\xc2\xf0\xd5j\xf7E\x8e"
    bool_0 = False
    maybe_0 = module_0.Maybe(bytes_0, bool_0)
    var_0 = maybe_0.to_try()
    int_0 = 134
    list_0 = [int_0, int_0, int_0, int_0]
    maybe_1 = module_0.Maybe(list_0, list_0)
    var_1 = maybe_1.to_lazy()
    var_2 = var_1.to_validation()
    var_2.filter(bytes_0)


def test_case_21():
    dict_0 = {}
    object_0 = module_2.object(**dict_0)
    maybe_0 = module_0.Maybe(dict_0, dict_0)
    var_0 = maybe_0.to_lazy()
    var_1 = maybe_0.to_lazy()
    var_2 = var_1.bind(object_0)
    var_2.to_validation()


def test_case_22():
    float_0 = -2416.28559
    maybe_0 = module_0.Maybe(float_0, float_0)
    var_0 = maybe_0.to_box()
    var_1 = maybe_0.bind(var_0)
    maybe_1 = module_0.Maybe(maybe_0, float_0)
    maybe_2 = module_0.Maybe(float_0, var_0)
    var_2 = maybe_2.map(var_0)
    var_3 = maybe_2.map(maybe_0)
    var_4 = var_3.ap(var_2)
    var_5 = var_1.bind(var_0)
    bool_0 = maybe_0.__eq__(maybe_2)
