# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.maybe as module_0
import typing as module_1


def test_case_0():
    str_0 = "@/y(?RvphwJZ?%a"
    maybe_0 = module_0.Maybe(str_0, str_0)


def test_case_1():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)


def test_case_2():
    int_0 = -2031
    int_1 = 2851
    maybe_0 = module_0.Maybe(int_1, int_1)
    bytes_0 = b"\xf8\xffD$DT\x81\xde\xf8\xb6\x8e\x992"
    dict_0 = {bytes_0: int_0, bytes_0: int_0}
    bool_0 = maybe_0.__eq__(dict_0)
    var_0 = maybe_0.to_either()
    var_0.to_either()


def test_case_3():
    int_0 = -952
    float_0 = -746.45044
    none_type_0 = None
    none_type_1 = None
    maybe_0 = module_0.Maybe(none_type_1, float_0)
    var_0 = maybe_0.map(none_type_0)
    var_1 = var_0.get_or_else(int_0)
    var_1.to_box()


def test_case_4():
    bool_0 = False
    none_type_0 = None
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    maybe_0.map(none_type_0)


def test_case_5():
    none_type_0 = None
    bool_0 = False
    bool_1 = True
    maybe_0 = module_0.Maybe(bool_1, bool_1)
    var_0 = maybe_0.bind(bool_0)
    var_1 = var_0.to_either()
    var_1.get_or_else(none_type_0)


def test_case_6():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    none_type_0 = None
    maybe_1 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_1.to_either()
    maybe_1.bind(var_0)


def test_case_7():
    bool_0 = False
    set_0 = {bool_0, bool_0, bool_0, bool_0}
    none_type_0 = None
    int_0 = -1220
    none_type_1 = None
    bool_1 = True
    maybe_0 = module_0.Maybe(none_type_1, bool_1)
    var_0 = maybe_0.ap(int_0)
    var_1 = var_0.to_either()
    var_2 = var_1.map(none_type_0)
    var_2.get_or_else(set_0)


def test_case_8():
    none_type_0 = None
    tuple_0 = ()
    maybe_0 = module_0.Maybe(tuple_0, tuple_0)
    maybe_1 = module_0.Maybe(none_type_0, maybe_0)
    var_0 = maybe_1.to_box()
    var_1 = maybe_1.filter(var_0)
    maybe_0.ap(none_type_0)


def test_case_9():
    dict_0 = {}
    maybe_0 = module_0.Maybe(dict_0, dict_0)
    bool_0 = maybe_0.__eq__(maybe_0)
    maybe_0.filter(maybe_0)


def test_case_10():
    none_type_0 = None
    tuple_0 = ()
    maybe_0 = module_0.Maybe(tuple_0, tuple_0)
    var_0 = maybe_0.to_either()
    maybe_1 = module_0.Maybe(none_type_0, maybe_0)
    var_1 = maybe_1.to_box()
    var_2 = maybe_1.filter(var_1)
    list_0 = [maybe_1, var_2]
    maybe_2 = module_0.Maybe(tuple_0, list_0)
    var_3 = maybe_1.get_or_else(tuple_0)
    var_4 = var_2.to_validation()


def test_case_11():
    bool_0 = False
    bytes_0 = b"\xe1\x0c\xcd\xbb\xde7,\xae\xcbj\x96L|\xfcA\x86\x9a"
    bool_1 = False
    maybe_0 = module_0.Maybe(bytes_0, bool_1)
    var_0 = maybe_0.get_or_else(bool_0)
    var_0.to_either()


def test_case_12():
    bool_0 = True
    none_type_0 = None
    bool_1 = False
    maybe_0 = module_0.Maybe(none_type_0, bool_1)
    var_0 = maybe_0.to_either()
    maybe_1 = module_0.Maybe(bool_0, bool_0)
    var_1 = maybe_1.to_try()


def test_case_13():
    str_0 = "a^Jb^&UO^#ta"
    tuple_0 = (str_0,)
    dict_0 = {tuple_0: str_0, tuple_0: tuple_0}
    list_0 = [dict_0]
    bool_0 = False
    maybe_0 = module_0.Maybe(list_0, bool_0)
    var_0 = maybe_0.to_box()
    var_0.to_box()


def test_case_14():
    bool_0 = True
    bytes_0 = b"{\xf1M\x82\xed"
    maybe_0 = module_0.Maybe(bool_0, bytes_0)
    var_0 = maybe_0.to_box()
    var_1 = maybe_0.to_validation()
    var_2 = var_1.to_lazy()
    maybe_1 = module_0.Maybe(bool_0, bool_0)
    var_3 = maybe_1.to_lazy()
    var_4 = maybe_1.to_box()


def test_case_15():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    bool_1 = False
    maybe_1 = module_0.Maybe(maybe_0, bool_1)
    var_0 = maybe_1.to_lazy()
    var_1 = var_0.to_box()
    var_2 = var_1.to_validation()
    str_0 = "\\e"
    bool_2 = True
    tuple_0 = (var_2, str_0, var_1, bool_2)
    bool_3 = False
    maybe_2 = module_0.Maybe(tuple_0, bool_3)


def test_case_16():
    none_type_0 = None
    bool_0 = False
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.to_try()
    var_0.ap(none_type_0)


def test_case_17():
    none_type_0 = None
    str_0 = ".K}_s8i2WTb"
    bool_0 = False
    maybe_0 = module_0.Maybe(str_0, bool_0)
    maybe_1 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_1.to_validation()
    var_0.to_validation()


def test_case_18():
    none_type_0 = None
    bool_0 = True
    set_0 = {none_type_0, none_type_0, bool_0}
    maybe_0 = module_0.Maybe(set_0, none_type_0)
    bool_1 = True
    maybe_1 = module_0.Maybe(bool_0, bool_1)
    var_0 = maybe_1.to_lazy()
    var_1 = var_0.to_validation()
    var_2 = var_1.to_either()
    var_3 = var_2.to_box()
    var_3.get_or_else(none_type_0)


def test_case_19():
    str_0 = "\n        :param semigroup: other semigroup to concat\n        :type semigroup: Max[B]\n        :returns: new Max with largest value\n        :rtype: Max[A | B]\n        "
    maybe_0 = module_0.Maybe(str_0, str_0)
    none_type_0 = None
    var_0 = maybe_0.map(none_type_0)
    var_1 = maybe_0.to_either()
    var_2 = maybe_0.ap(none_type_0)
    var_3 = var_2.to_lazy()
    bool_0 = maybe_0.__eq__(var_3)
    var_4 = maybe_0.to_lazy()
    var_5 = maybe_0.to_try()
    var_6 = var_0.filter(var_3)
    bool_1 = var_5.__eq__(var_4)
    var_7 = var_2.ap(maybe_0)
    bool_2 = var_0.__eq__(var_7)
    var_8 = maybe_0.get_or_else(maybe_0)
    var_9 = maybe_0.to_box()
    var_10 = var_8.to_try()
    var_11 = var_2.get_or_else(var_4)
    var_12 = maybe_0.filter(var_11)
    maybe_1 = module_0.Maybe(var_2, str_0)
    var_13 = maybe_1.to_box()
    var_9.map(var_1)


def test_case_20():
    str_0 = "\n        :param semigroup: other semigroup to concat\n        :type semigroup: Max[B]\n        :returns: new Max with largest value\n        :rtype: Max[A | B]\n        "
    maybe_0 = module_0.Maybe(str_0, str_0)
    var_0 = maybe_0.to_either()
    var_1 = maybe_0.ap(var_0)
    var_2 = var_1.to_lazy()
    bool_0 = maybe_0.__eq__(var_2)
    var_3 = var_1.ap(var_1)
    var_4 = maybe_0.to_lazy()
    maybe_1 = module_0.Maybe(var_0, maybe_0)
    maybe_2 = module_0.Maybe(var_0, var_1)
    var_5 = var_0.to_try()
    generic_0 = module_1.Generic()
    bool_1 = maybe_2.__eq__(maybe_1)
    var_6 = maybe_0.ap(var_1)
    var_0.get_or_else(var_6)
