# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.maybe as module_0


def test_case_0():
    str_0 = "!rcSTh?3"
    maybe_0 = module_0.Maybe(str_0, str_0)


def test_case_1():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)


def test_case_2():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    bool_0 = maybe_0.__eq__(maybe_0)
    maybe_0.filter(none_type_0)


def test_case_3():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    bool_1 = maybe_0.__eq__(bool_0)


def test_case_4():
    int_0 = -179
    maybe_0 = module_0.Maybe(int_0, int_0)
    bool_0 = maybe_0.__eq__(int_0)
    str_0 = "\n        Returns successful Validation with value and empty errors list.\n\n        :params value: value to store 4n Validation\n        :type value: A\n        :returns: Successful Validation\n        :rtype: Validation[A, []\n        "
    bool_1 = False
    maybe_1 = module_0.Maybe(str_0, bool_1)
    var_0 = maybe_0.to_lazy()
    var_1 = maybe_0.filter(var_0)
    var_2 = var_1.map(var_0)
    var_3 = maybe_1.ap(var_0)
    bool_2 = False
    maybe_1.filter(bool_2)


def test_case_5():
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.filter(none_type_0)
    var_1 = var_0.to_box()
    int_0 = 859
    var_2 = maybe_0.get_or_else(int_0)
    var_3 = var_1.to_validation()
    var_4 = var_3.to_lazy()
    var_5 = var_0.to_try()
    maybe_1 = module_0.Maybe(none_type_0, none_type_0)
    maybe_2 = module_0.Maybe(none_type_0, none_type_0)
    maybe_1.map(none_type_0)


def test_case_6():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    bool_0 = False
    maybe_1 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_1.get_or_else(maybe_0)
    bool_1 = False
    list_0 = [bool_1, bool_1, bool_1, bool_1]
    maybe_2 = module_0.Maybe(list_0, list_0)
    var_1 = maybe_2.bind(var_0)


def test_case_7():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.to_lazy()
    maybe_0.bind(var_0)


def test_case_8():
    bool_0 = True
    int_0 = -1321
    maybe_0 = module_0.Maybe(int_0, bool_0)
    var_0 = maybe_0.ap(bool_0)
    var_1 = maybe_0.to_validation()
    var_2 = var_1.to_box()
    var_3 = var_2.to_either()
    var_4 = var_0.to_lazy()
    var_5 = var_4.ap(int_0)
    maybe_1 = module_0.Maybe(int_0, int_0)
    var_6 = maybe_1.filter(bool_0)
    str_0 = "Dt_`.);@wdN!\\CCwbS"
    maybe_2 = module_0.Maybe(str_0, str_0)
    str_0.get_or_else(var_6)


def test_case_9():
    int_0 = -179
    maybe_0 = module_0.Maybe(int_0, int_0)
    bool_0 = maybe_0.__eq__(int_0)
    var_0 = maybe_0.to_try()
    maybe_1 = module_0.Maybe(bool_0, bool_0)
    var_1 = maybe_0.to_lazy()
    var_2 = maybe_0.filter(var_1)
    var_3 = maybe_1.ap(var_1)
    bool_1 = False
    maybe_1.filter(bool_1)


def test_case_10():
    int_0 = -179
    maybe_0 = module_0.Maybe(int_0, int_0)
    bool_0 = maybe_0.to_lazy()
    str_0 = "\n        Returns successful Validation with value and empty errors list.\n\n        :params value: value to store 4n Validation\n        :type value: A\n        :returns: Successful Validation\n        :rtype: Validation[A, []\n        "
    bool_1 = False
    maybe_1 = module_0.Maybe(str_0, bool_1)
    var_0 = maybe_0.to_lazy()
    var_1 = maybe_0.filter(var_0)
    bool_2 = False
    maybe_1.filter(bool_2)


def test_case_11():
    bytes_0 = b"\xe6s\xec\xa3\xea\x15\x80ZR\xf4/\x9e\x11\xbb\x82\x87\x12"
    maybe_0 = module_0.Maybe(bytes_0, bytes_0)
    var_0 = maybe_0.to_either()
    var_1 = maybe_0.to_box()
    var_2 = maybe_0.get_or_else(bytes_0)
    var_3 = var_0.to_lazy()
    var_4 = var_1.to_lazy()
    bool_0 = var_3.__eq__(maybe_0)
    bool_1 = False
    maybe_1 = module_0.Maybe(var_4, bool_1)


def test_case_12():
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, none_type_0)
    var_0 = maybe_0.to_either()
    var_0.filter(none_type_0)


def test_case_13():
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.to_box()
    var_1 = var_0.to_try()


def test_case_14():
    int_0 = 0
    maybe_0 = module_0.Maybe(int_0, int_0)
    var_0 = maybe_0.to_box()
    var_0.map(var_0)


def test_case_15():
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.to_lazy()
    var_0.get_or_else(none_type_0)


def test_case_16():
    str_0 = "\x0b[j\x0cQD_fH%!ax}9=*"
    int_0 = -872
    bool_0 = False
    maybe_0 = module_0.Maybe(int_0, bool_0)
    var_0 = maybe_0.to_lazy()
    var_1 = var_0.ap(str_0)


def test_case_17():
    none_type_0 = None
    bool_0 = False
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.to_try()
    var_0.to_either()


def test_case_18():
    none_type_0 = None
    bool_0 = False
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.to_validation()


def test_case_19():
    int_0 = -179
    maybe_0 = module_0.Maybe(int_0, int_0)
    var_0 = maybe_0.to_lazy()
    var_1 = var_0.to_either()
    bool_0 = False
    maybe_1 = module_0.Maybe(var_1, bool_0)
    var_2 = maybe_0.to_lazy()
    var_3 = maybe_0.to_lazy()
    var_4 = maybe_0.filter(var_3)
    var_5 = maybe_1.get_or_else(bool_0)
    bool_1 = False
    maybe_1.filter(bool_1)


def test_case_20():
    none_type_0 = None
    bool_0 = False
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.to_lazy()
    set_0 = set()
    var_1 = maybe_0.to_validation()
    bool_1 = var_0.__eq__(maybe_0)
    var_2 = var_0.to_validation()
    list_0 = []
    maybe_1 = module_0.Maybe(set_0, list_0)
    var_3 = var_0.to_either()
    set_0.to_box()


def test_case_21():
    none_type_0 = None
    float_0 = 374.267486
    maybe_0 = module_0.Maybe(float_0, float_0)
    var_0 = maybe_0.bind(none_type_0)
    bool_0 = maybe_0.__eq__(var_0)
    var_1 = var_0.to_either()
    var_2 = maybe_0.ap(none_type_0)
    var_3 = var_1.to_try()
    var_4 = maybe_0.bind(var_3)
    var_5 = var_1.to_lazy()
    var_3.to_try()


def test_case_22():
    int_0 = -80
    maybe_0 = module_0.Maybe(int_0, int_0)
    bool_0 = maybe_0.__eq__(maybe_0)
