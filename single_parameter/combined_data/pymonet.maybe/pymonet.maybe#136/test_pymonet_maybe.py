# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.maybe as module_0
import builtins as module_1


def test_case_0():
    int_0 = -310
    maybe_0 = module_0.Maybe(int_0, int_0)


def test_case_1():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)


def test_case_2():
    none_type_0 = None
    bool_0 = True
    str_0 = "Dq+!}Dd"
    maybe_0 = module_0.Maybe(bool_0, str_0)
    var_0 = maybe_0.to_lazy()
    object_0 = module_1.object()
    maybe_1 = module_0.Maybe(var_0, object_0)
    bool_1 = maybe_0.__eq__(none_type_0)


def test_case_3():
    bytes_0 = b"\x86\x02\x91\x98*\xb0H\xee\xe6NI"
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.map(bytes_0)
    var_1 = var_0.to_either()
    var_2 = var_0.map(bool_0)
    var_3 = var_0.filter(var_2)
    var_4 = var_2.to_lazy()
    var_5 = var_3.to_try()
    var_6 = var_4.bind(var_4)
    var_7 = var_1.to_validation()


def test_case_4():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    maybe_0.map(maybe_0)


def test_case_5():
    complex_0 = -1065.489 - 2206.874231j
    dict_0 = {complex_0: complex_0}
    bool_0 = False
    set_0 = {bool_0}
    maybe_0 = module_0.Maybe(set_0, bool_0)
    bool_1 = maybe_0.__eq__(dict_0)
    bool_2 = False
    bytes_0 = b"\xda\xd6\x8f"
    bool_3 = True
    maybe_1 = module_0.Maybe(bytes_0, bool_3)
    var_0 = maybe_1.bind(bool_2)


def test_case_6():
    complex_0 = -1065.489 - 2206.874231j
    dict_0 = {complex_0: complex_0}
    bool_0 = False
    set_0 = {bool_0}
    maybe_0 = module_0.Maybe(set_0, bool_0)
    bool_1 = maybe_0.__eq__(dict_0)
    bool_2 = False
    maybe_0.bind(bool_2)


def test_case_7():
    bool_0 = True
    str_0 = "Dq+!}Dd"
    maybe_0 = module_0.Maybe(bool_0, str_0)
    var_0 = maybe_0.to_lazy()
    bool_1 = True
    maybe_1 = module_0.Maybe(str_0, bool_1)
    var_1 = maybe_1.map(var_0)
    var_2 = maybe_1.to_try()
    var_3 = var_1.ap(bool_1)
    var_4 = maybe_0.to_box()
    bool_2 = var_3.__eq__(var_0)


def test_case_8():
    bool_0 = True
    bytes_0 = b"ZaG\xb7\xab\x9b\x0bYB\xbb \xcb\x06\x7fy"
    none_type_0 = None
    maybe_0 = module_0.Maybe(bytes_0, none_type_0)
    maybe_0.ap(bool_0)


def test_case_9():
    bytes_0 = b"\xc0\xbaeXI"
    float_0 = -1887.6692
    set_0 = {float_0}
    maybe_0 = module_0.Maybe(set_0, float_0)
    var_0 = maybe_0.filter(bytes_0)
    none_type_0 = None
    maybe_1 = module_0.Maybe(bytes_0, none_type_0)
    maybe_1.filter(maybe_1)


def test_case_10():
    complex_0 = -450.171 - 747.63436j
    none_type_0 = None
    maybe_0 = module_0.Maybe(complex_0, complex_0)
    var_0 = maybe_0.to_box()
    maybe_1 = module_0.Maybe(complex_0, complex_0)
    var_1 = maybe_1.bind(var_0)
    var_2 = var_1.get_or_else(none_type_0)
    bool_0 = True
    tuple_0 = (complex_0, var_2, bool_0)
    list_0 = [tuple_0, var_2]
    str_0 = "wyU?X3tnu`"
    bool_1 = True
    maybe_2 = module_0.Maybe(str_0, bool_1)
    var_3 = maybe_2.ap(list_0)


def test_case_11():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    bool_0 = maybe_0.__eq__(maybe_0)
    var_0 = maybe_0.get_or_else(none_type_0)
    maybe_0.map(maybe_0)


def test_case_12():
    bytes_0 = b"\x95\x8e\xae\x19%!\x17"
    float_0 = 842.02
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.to_either()
    var_1 = var_0.ap(float_0)
    var_2 = var_1.to_validation()
    var_2.ap(bytes_0)


def test_case_13():
    none_type_0 = None
    bool_0 = True
    bool_1 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_1)
    bool_2 = maybe_0.__eq__(bool_0)
    maybe_1 = module_0.Maybe(maybe_0, bool_2)
    var_0 = maybe_1.to_either()
    var_1 = maybe_0.filter(var_0)
    var_2 = var_1.to_validation()
    var_3 = maybe_0.to_box()
    var_3.map(var_2)


def test_case_14():
    bool_0 = True
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.to_box()
    var_0.get_or_else(bool_0)


def test_case_15():
    none_type_0 = None
    bool_0 = False
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.to_lazy()
    object_0 = module_1.object()
    maybe_1 = module_0.Maybe(none_type_0, var_0)
    bool_1 = maybe_0.__eq__(bool_0)


def test_case_16():
    int_0 = -310
    maybe_0 = module_0.Maybe(int_0, int_0)
    var_0 = maybe_0.to_try()


def test_case_17():
    bool_0 = True
    bool_1 = False
    maybe_0 = module_0.Maybe(bool_0, bool_1)
    var_0 = maybe_0.to_try()
    maybe_0.filter(bool_0)


def test_case_18():
    bytes_0 = b"Q\x08\xf1iB\x94"
    maybe_0 = module_0.Maybe(bytes_0, bytes_0)
    maybe_1 = module_0.Maybe(maybe_0, maybe_0)
    var_0 = maybe_1.to_validation()
    bool_0 = False
    maybe_2 = module_0.Maybe(maybe_0, bool_0)
    maybe_2.bind(var_0)


def test_case_19():
    set_0 = set()
    none_type_0 = None
    bool_0 = False
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.to_validation()
    var_0.ap(set_0)


def test_case_20():
    bytes_0 = b"\x86\x02\x91\x98*\xb0H\xee\xe6NI"
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.map(bytes_0)
    var_1 = var_0.to_lazy()
    var_2 = var_0.map(bool_0)
    var_3 = var_0.filter(var_2)
    var_4 = var_2.to_lazy()
    var_5 = var_3.to_try()
    var_6 = var_4.bind(var_4)
    var_7 = var_1.to_validation()


def test_case_21():
    bool_0 = True
    str_0 = "Dq+!}Dd"
    maybe_0 = module_0.Maybe(bool_0, str_0)
    var_0 = maybe_0.to_lazy()
    object_0 = module_1.object()
    maybe_1 = module_0.Maybe(var_0, bool_0)
    bool_1 = maybe_0.__eq__(maybe_0)


def test_case_22():
    str_0 = "AB?]XVT,UjB>Ft$`,F\r%"
    bool_0 = False
    maybe_0 = module_0.Maybe(str_0, bool_0)
    var_0 = maybe_0.to_lazy()
    var_1 = var_0.to_try()
    var_1.to_try()


def test_case_23():
    none_type_0 = None
    bool_0 = False
    bool_1 = True
    maybe_0 = module_0.Maybe(bool_0, bool_1)
    var_0 = maybe_0.map(none_type_0)
    var_1 = var_0.to_either()
    str_0 = "\n        Transform Lazy into Either (Right) with constructor_fn result.\n\n        :returns: Right monad with constructor_fn result\n        :rtype: Right[A]\n        "
    none_type_1 = None
    maybe_1 = module_0.Maybe(str_0, none_type_1)
    var_2 = var_0.map(none_type_1)
    var_3 = maybe_0.to_lazy()
    object_0 = module_1.object()
    maybe_2 = module_0.Maybe(var_0, var_1)
    bool_2 = maybe_0.__eq__(maybe_2)
