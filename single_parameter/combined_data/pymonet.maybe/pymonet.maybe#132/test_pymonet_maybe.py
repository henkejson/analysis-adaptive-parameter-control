# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.maybe as module_0


def test_case_0():
    str_0 = "(K{]'d."
    maybe_0 = module_0.Maybe(str_0, str_0)


def test_case_1():
    bool_0 = False
    maybe_0 = module_0.Maybe(bool_0, bool_0)


def test_case_2():
    dict_0 = {}
    bool_0 = True
    none_type_0 = None
    maybe_0 = module_0.Maybe(dict_0, bool_0)
    var_0 = maybe_0.filter(dict_0)
    var_1 = var_0.to_validation()
    var_2 = maybe_0.map(var_1)
    var_3 = var_2.to_lazy()
    bool_1 = maybe_0.__eq__(bool_0)
    var_4 = var_3.to_either()
    var_5 = var_4.to_try()
    var_5.ap(none_type_0)


def test_case_3():
    float_0 = 127.99994
    maybe_0 = module_0.Maybe(float_0, float_0)
    var_0 = maybe_0.get_or_else(float_0)
    var_1 = maybe_0.map(var_0)
    bool_0 = False
    var_2 = maybe_0.to_try()
    maybe_1 = module_0.Maybe(float_0, bool_0)
    var_0.map(var_2)


def test_case_4():
    bool_0 = False
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    maybe_1 = module_0.Maybe(bool_0, none_type_0)
    var_0 = maybe_0.to_lazy()
    var_1 = maybe_1.to_lazy()
    bool_1 = var_1.__eq__(bool_0)
    maybe_2 = module_0.Maybe(none_type_0, none_type_0)
    maybe_2.ap(maybe_2)


def test_case_5():
    bool_0 = False
    str_0 = "\n        Returns True when errors list are empty.\n\n        :returns: True for empty errors list\n        :rtype: Boolean\n        "
    maybe_0 = module_0.Maybe(str_0, str_0)
    var_0 = maybe_0.bind(bool_0)
    var_1 = maybe_0.to_either()
    var_2 = maybe_0.to_validation()
    var_2.bind(bool_0)


def test_case_6():
    bool_0 = True
    bool_1 = True
    maybe_0 = module_0.Maybe(bool_0, bool_1)
    var_0 = maybe_0.ap(maybe_0)
    var_1 = var_0.to_try()
    var_2 = maybe_0.to_either()
    var_3 = var_0.to_validation()
    var_4 = maybe_0.to_validation()


def test_case_7():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.to_try()
    bool_0 = maybe_0.__eq__(maybe_0)
    maybe_0.ap(var_0)


def test_case_8():
    int_0 = 726
    str_0 = "I( nf\rj"
    complex_0 = 373.12722 - 5532.106236j
    maybe_0 = module_0.Maybe(str_0, complex_0)
    var_0 = maybe_0.filter(int_0)
    str_1 = "\n        Returns True when errors list are empty.\n\n        :returns: True for empty errors list\n        :rtype: Boolean\n        "
    maybe_1 = module_0.Maybe(str_1, str_1)


def test_case_9():
    bool_0 = False
    str_0 = "\n        Returns True when errors list are empty.\n\n        :returns: True for empty errors list\n        :rtype: Boolean\n        "
    maybe_0 = module_0.Maybe(str_0, str_0)
    var_0 = maybe_0.get_or_else(bool_0)
    var_1 = maybe_0.bind(var_0)
    var_2 = var_1.to_either()
    str_0.bind(bool_0)


def test_case_10():
    int_0 = 425
    list_0 = [int_0]
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.get_or_else(list_0)
    complex_0 = 161.60856 - 119.37j
    maybe_1 = module_0.Maybe(complex_0, complex_0)
    var_1 = maybe_1.filter(complex_0)
    var_2 = maybe_1.to_validation()
    var_3 = var_1.to_try()
    var_3.to_either()


def test_case_11():
    none_type_0 = None
    none_type_1 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_1, bool_0)
    var_0 = maybe_0.to_either()
    var_0.get_or_else(none_type_0)


def test_case_12():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.to_validation()
    var_1 = maybe_0.to_either()
    var_2 = var_1.to_box()
    var_3 = var_2.to_validation()
    bool_0 = var_1.__eq__(var_1)
    var_3.map(none_type_0)


def test_case_13():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    bool_1 = True
    maybe_1 = module_0.Maybe(maybe_0, bool_1)
    var_0 = maybe_1.to_box()


def test_case_14():
    bool_0 = False
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.to_validation()
    var_1 = var_0.to_lazy()
    int_0 = -620
    set_0 = {int_0, int_0}
    bool_1 = False
    maybe_1 = module_0.Maybe(int_0, bool_1)
    bool_2 = maybe_1.__eq__(bool_1)
    var_2 = maybe_1.to_box()
    var_3 = var_2.to_validation()
    bool_3 = var_2.__eq__(int_0)
    set_0.to_validation()


def test_case_15():
    int_0 = 0
    bytes_0 = b"\xa6I!\xaf\xd2\x10\xd8\xfati`4\xd9\r\x85\x9d\nU"
    maybe_0 = module_0.Maybe(int_0, bytes_0)
    var_0 = maybe_0.to_lazy()


def test_case_16():
    str_0 = ","
    complex_0 = -651.61416 + 1844j
    bool_0 = False
    maybe_0 = module_0.Maybe(complex_0, bool_0)
    var_0 = maybe_0.to_lazy()
    var_0.get_or_else(str_0)


def test_case_17():
    str_0 = "K"
    maybe_0 = module_0.Maybe(str_0, str_0)
    var_0 = maybe_0.to_try()
    var_0.to_either()


def test_case_18():
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.to_lazy()
    var_1 = var_0.to_try()
    var_1.to_box()


def test_case_19():
    bool_0 = True
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    maybe_1 = module_0.Maybe(bool_0, none_type_0)
    var_0 = maybe_0.filter(none_type_0)
    var_1 = maybe_1.to_lazy()
    bool_1 = var_1.to_box()
    maybe_2 = module_0.Maybe(none_type_0, none_type_0)
    maybe_2.ap(maybe_2)


def test_case_20():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.to_either()
    maybe_0.bind(maybe_0)


def test_case_21():
    bool_0 = True
    bool_1 = True
    maybe_0 = module_0.Maybe(bool_0, bool_1)
    var_0 = maybe_0.ap(maybe_0)
    var_1 = var_0.filter(var_0)
    bool_2 = var_0.__eq__(var_1)
    var_2 = var_0.to_try()
    var_3 = var_0.filter(maybe_0)
    var_4 = var_0.to_validation()
    bool_3 = var_4.__eq__(bool_1)


def test_case_22():
    bool_0 = True
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    maybe_1 = module_0.Maybe(bool_0, none_type_0)
    var_0 = maybe_0.filter(none_type_0)
    var_1 = maybe_1.to_lazy()
    maybe_1.filter(none_type_0)


def test_case_23():
    bool_0 = True
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    maybe_1 = module_0.Maybe(bool_0, none_type_0)
    var_0 = maybe_0.filter(none_type_0)
    var_1 = maybe_1.to_lazy()
    var_2 = var_0.bind(var_1)
    bool_1 = maybe_1.__eq__(var_0)
    var_3 = var_0.to_box()
    var_4 = var_2.to_either()
    var_5 = var_3.to_either()
    var_6 = maybe_0.map(maybe_0)
