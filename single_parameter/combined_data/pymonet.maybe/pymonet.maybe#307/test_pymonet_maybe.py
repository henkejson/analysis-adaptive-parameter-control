# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.maybe as module_0


def test_case_0():
    str_0 = 'O"I9n:P0a'
    maybe_0 = module_0.Maybe(str_0, str_0)


def test_case_1():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)


def test_case_2():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.to_lazy()
    bool_0 = maybe_0.__eq__(none_type_0)
    var_1 = var_0.to_try()
    var_2 = maybe_0.to_validation()
    var_2.get_or_else(bool_0)


def test_case_3():
    str_0 = "c"
    maybe_0 = module_0.Maybe(str_0, str_0)
    var_0 = maybe_0.to_validation()
    none_type_0 = None
    var_1 = maybe_0.filter(none_type_0)
    var_2 = maybe_0.ap(str_0)
    var_3 = maybe_0.to_box()
    var_4 = var_1.map(var_0)
    var_5 = var_2.to_lazy()
    var_6 = var_5.to_try()


def test_case_4():
    bool_0 = True
    none_type_0 = None
    maybe_0 = module_0.Maybe(bool_0, none_type_0)
    var_0 = maybe_0.to_lazy()
    var_1 = maybe_0.to_lazy()
    var_2 = var_1.to_validation()
    maybe_0.map(bool_0)


def test_case_5():
    float_0 = 1130.9933036309562
    float_1 = 2309.22979
    bool_0 = True
    maybe_0 = module_0.Maybe(float_0, float_1)
    var_0 = maybe_0.get_or_else(maybe_0)
    maybe_1 = module_0.Maybe(var_0, bool_0)
    var_1 = maybe_1.get_or_else(float_1)
    var_2 = maybe_0.bind(var_1)
    var_3 = maybe_0.filter(float_0)
    var_4 = maybe_0.to_lazy()
    bool_1 = False
    bool_2 = False
    maybe_2 = module_0.Maybe(bool_1, bool_2)
    maybe_2.filter(var_0)


def test_case_6():
    none_type_0 = None
    list_0 = []
    list_1 = [list_0, list_0]
    maybe_0 = module_0.Maybe(list_1, list_0)
    maybe_0.bind(none_type_0)


def test_case_7():
    bool_0 = False
    str_0 = 'O"I9n:P0a'
    maybe_0 = module_0.Maybe(str_0, str_0)
    var_0 = maybe_0.ap(bool_0)


def test_case_8():
    str_0 = "\n        Returns successful Validation with value and empty errors list.\n\n        :params value: value to store in Validation\n        :type value: A\n        :returns: Successful Validation\n        :rtype: Validation[A, []]\n        "
    tuple_0 = (str_0,)
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    bool_0 = True
    maybe_1 = module_0.Maybe(tuple_0, bool_0)
    var_0 = maybe_0.get_or_else(tuple_0)
    tuple_1 = ()
    maybe_2 = maybe_1.to_validation()
    maybe_3 = module_0.Maybe(str_0, maybe_0)
    var_1 = maybe_3.filter(maybe_1)
    var_2 = maybe_0.to_try()
    var_3 = maybe_3.to_validation()
    maybe_0.ap(tuple_1)


def test_case_9():
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.filter(maybe_0)
    bool_1 = False
    maybe_1 = module_0.Maybe(none_type_0, bool_1)
    maybe_1.filter(bool_1)


def test_case_10():
    float_0 = 2309.22979
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.get_or_else(float_0)
    var_1 = maybe_0.filter(maybe_0)
    bool_1 = False
    maybe_1 = module_0.Maybe(var_0, bool_1)
    maybe_1.filter(bool_1)


def test_case_11():
    bytes_0 = b"-\x88rm\xa8\x86\x14\xbd\xb2\x89"
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.get_or_else(none_type_0)
    var_1 = maybe_0.to_validation()
    bool_0 = True
    maybe_1 = module_0.Maybe(bytes_0, bool_0)
    var_2 = maybe_1.to_box()
    var_2.map(bytes_0)


def test_case_12():
    bytes_0 = b"L\xfa/M\xbd\xe5\x83z<\x0fx\x13\xe6y:\x11{\xd4"
    float_0 = -5855.26485
    bool_0 = True
    maybe_0 = module_0.Maybe(float_0, bool_0)
    bool_1 = maybe_0.__eq__(maybe_0)
    bool_2 = True
    maybe_1 = module_0.Maybe(bool_2, bool_2)
    var_0 = maybe_1.map(float_0)
    var_1 = var_0.to_either()
    var_1.filter(bytes_0)


def test_case_13():
    set_0 = set()
    maybe_0 = module_0.Maybe(set_0, set_0)
    var_0 = maybe_0.to_either()
    var_1 = var_0.to_lazy()
    var_2 = var_1.to_box()
    bool_0 = var_1.__eq__(var_0)
    var_1.get_or_else(set_0)


def test_case_14():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.to_box()
    maybe_0.filter(var_0)


def test_case_15():
    complex_0 = -938.086376 + 2268.1j
    maybe_0 = module_0.Maybe(complex_0, complex_0)
    var_0 = maybe_0.to_lazy()
    var_1 = var_0.to_validation()
    var_2 = var_1.to_either()
    var_3 = var_2.to_validation()


def test_case_16():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.to_either()
    var_1 = var_0.ap(var_0)
    var_2 = maybe_0.to_try()
    var_0.get_or_else(maybe_0)


def test_case_17():
    set_0 = set()
    str_0 = "\n    Max is a Monoid that will combines 2 numbers, resulting in the largest of the two.\n    "
    bool_0 = False
    maybe_0 = module_0.Maybe(set_0, bool_0)
    var_0 = maybe_0.to_try()
    var_1 = var_0.get_or_else(str_0)
    var_1.to_box()


def test_case_18():
    float_0 = 1129.903
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.to_validation()
    var_1 = maybe_0.to_try()
    var_2 = maybe_0.ap(float_0)
    maybe_1 = module_0.Maybe(float_0, var_2)
    var_3 = var_2.ap(none_type_0)
    bool_1 = maybe_1.__eq__(var_2)
    var_4 = var_1.get_or_else(maybe_1)
    var_5 = maybe_0.ap(none_type_0)
    maybe_2 = module_0.Maybe(var_5, bool_0)
    var_6 = maybe_2.get_or_else(float_0)
    var_7 = var_3.bind(var_4)
    bool_2 = var_2.__eq__(var_7)
    var_8 = var_7.to_validation()
    var_9 = maybe_0.filter(var_2)
    maybe_3 = module_0.Maybe(var_6, var_6)
    var_10 = maybe_3.to_lazy()
    var_11 = var_7.filter(var_1)
    var_12 = var_9.to_lazy()
    bool_3 = maybe_2.__eq__(var_10)
    var_13 = var_7.bind(float_0)
    var_14 = var_2.to_lazy()
    var_10.to_lazy()


def test_case_19():
    bytes_0 = b"\xfe<H\xf5\r\xc15\xdfQ\x8c\x93Vx\x90\x91"
    float_0 = -5855.438957575195
    bool_0 = False
    maybe_0 = module_0.Maybe(float_0, bool_0)
    bool_1 = maybe_0.__eq__(maybe_0)
    bool_2 = True
    maybe_1 = module_0.Maybe(bool_2, bool_2)
    var_0 = maybe_1.map(float_0)
    var_1 = var_0.ap(float_0)
    var_2 = var_1.filter(bytes_0)
