# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.maybe as module_0


def test_case_0():
    str_0 = "ImmutableList[T]"
    maybe_0 = module_0.Maybe(str_0, str_0)


def test_case_1():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)


def test_case_2():
    bool_0 = True
    set_0 = {bool_0, bool_0}
    maybe_0 = module_0.Maybe(set_0, bool_0)
    var_0 = maybe_0.filter(bool_0)
    bool_1 = maybe_0.__eq__(var_0)
    var_1 = var_0.map(maybe_0)
    bool_2 = False
    maybe_1 = module_0.Maybe(bool_0, bool_2)
    maybe_1.filter(var_1)


def test_case_3():
    int_0 = -2291
    set_0 = {int_0}
    maybe_0 = module_0.Maybe(int_0, set_0)
    bool_0 = True
    maybe_1 = module_0.Maybe(maybe_0, bool_0)
    var_0 = maybe_0.map(set_0)
    maybe_2 = module_0.Maybe(maybe_1, bool_0)
    var_1 = maybe_2.filter(maybe_2)


def test_case_4():
    str_0 = "$byi>$kJ_Fp.ac`6]bSP"
    bool_0 = False
    maybe_0 = module_0.Maybe(str_0, bool_0)
    maybe_0.map(str_0)


def test_case_5():
    bool_0 = True
    str_0 = "ImmutableList[T]"
    maybe_0 = module_0.Maybe(str_0, str_0)
    var_0 = maybe_0.bind(bool_0)
    var_1 = var_0.to_validation()


def test_case_6():
    int_0 = -5065
    bool_0 = False
    maybe_0 = module_0.Maybe(int_0, bool_0)
    set_0 = set()
    bool_1 = False
    none_type_0 = None
    maybe_1 = module_0.Maybe(none_type_0, set_0)
    var_0 = maybe_1.to_try()
    var_1 = maybe_1.to_try()
    var_2 = maybe_1.to_either()
    maybe_2 = module_0.Maybe(set_0, bool_1)
    maybe_2.bind(none_type_0)


def test_case_7():
    int_0 = 1306
    bool_0 = True
    set_0 = {int_0, int_0}
    maybe_0 = module_0.Maybe(int_0, bool_0)
    var_0 = maybe_0.ap(set_0)
    var_1 = var_0.to_box()
    var_2 = var_1.to_validation()
    var_3 = var_2.to_lazy()
    maybe_1 = module_0.Maybe(int_0, bool_0)
    var_4 = maybe_1.bind(int_0)
    var_5 = maybe_1.to_validation()


def test_case_8():
    bytes_0 = b""
    dict_0 = {}
    bool_0 = False
    maybe_0 = module_0.Maybe(dict_0, bool_0)
    maybe_0.ap(bytes_0)


def test_case_9():
    float_0 = -2193.837667
    bool_0 = True
    maybe_0 = module_0.Maybe(float_0, bool_0)
    var_0 = maybe_0.to_either()
    var_1 = maybe_0.filter(var_0)


def test_case_10():
    float_0 = 3448.0
    bool_0 = False
    maybe_0 = module_0.Maybe(float_0, bool_0)
    var_0 = maybe_0.to_either()
    maybe_0.filter(var_0)


def test_case_11():
    bytes_0 = b"\x91"
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.get_or_else(bytes_0)


def test_case_12():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.to_lazy()
    var_1 = maybe_0.get_or_else(maybe_0)
    var_2 = maybe_0.get_or_else(none_type_0)
    maybe_0.filter(var_2)


def test_case_13():
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.to_either()


def test_case_14():
    bytes_0 = b"\xebt~\xff\xf9/\x05\xa0b\xc4Q\xd8Q**"
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.to_box()
    var_1 = var_0.to_try()
    var_1.filter(bytes_0)


def test_case_15():
    str_0 = "foXdD!T\x0b0\x0b;vM "
    set_0 = {str_0, str_0, str_0}
    maybe_0 = module_0.Maybe(set_0, set_0)
    var_0 = maybe_0.to_lazy()


def test_case_16():
    list_0 = []
    none_type_0 = None
    maybe_0 = module_0.Maybe(list_0, none_type_0)
    var_0 = maybe_0.to_lazy()
    var_0.to_lazy()


def test_case_17():
    bool_0 = True
    int_0 = 768
    bool_1 = True
    maybe_0 = module_0.Maybe(int_0, bool_1)
    var_0 = maybe_0.bind(bool_0)
    var_1 = var_0.to_try()


def test_case_18():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.to_try()
    var_0.to_lazy()


def test_case_19():
    bool_0 = False
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.to_validation()
    maybe_0.filter(var_0)


def test_case_20():
    bool_0 = True
    none_type_0 = None
    bool_1 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_1)
    var_0 = maybe_0.to_lazy()
    var_1 = var_0.to_either()
    var_2 = var_1.to_validation()
    var_2.ap(bool_0)


def test_case_21():
    int_0 = 1306
    bool_0 = True
    maybe_0 = module_0.Maybe(int_0, bool_0)
    var_0 = maybe_0.to_validation()
    bool_1 = maybe_0.__eq__(bool_0)


def test_case_22():
    int_0 = 1012
    none_type_0 = None
    bool_0 = False
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.to_try()
    var_1 = var_0.get_or_else(int_0)
    none_type_1 = None
    maybe_1 = module_0.Maybe(none_type_1, none_type_1)
    bool_1 = False
    maybe_2 = module_0.Maybe(none_type_1, bool_1)
    bool_2 = maybe_1.__eq__(maybe_1)
    bool_3 = maybe_1.__eq__(int_0)
    set_0 = {int_0, int_0, int_0, int_0}
    tuple_0 = (int_0, set_0)
    bool_4 = True
    maybe_3 = module_0.Maybe(tuple_0, bool_4)
    var_2 = maybe_1.to_box()


def test_case_23():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.to_lazy()
    var_1 = var_0.to_validation()
    var_2 = maybe_0.get_or_else(maybe_0)
    var_3 = var_0.to_try()
    var_4 = maybe_0.get_or_else(none_type_0)
    maybe_0.filter(var_4)


def test_case_24():
    none_type_0 = None
    int_0 = 436
    maybe_0 = module_0.Maybe(none_type_0, int_0)
    var_0 = maybe_0.filter(int_0)
    var_1 = maybe_0.get_or_else(int_0)
    maybe_1 = module_0.Maybe(maybe_0, var_1)
    var_2 = maybe_1.to_box()
    var_3 = maybe_0.bind(var_1)
    var_4 = maybe_1.to_try()
    var_5 = var_3.to_validation()
    var_6 = var_3.ap(var_2)
    bool_0 = var_6.__eq__(maybe_1)
    var_4.to_validation()
