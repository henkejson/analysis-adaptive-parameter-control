# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.maybe as module_0


def test_case_0():
    str_0 = "wla"
    maybe_0 = module_0.Maybe(str_0, str_0)


def test_case_1():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)


def test_case_2():
    str_0 = "\n        If Maybe is empty return new empty Maybe, in other case\n        takes mapper function and returns result of mapper.\n\n        :param mapper: function to call with Maybe.value\n        :type mapper: Function(A) -> Maybe[B]\n        :returns: Maybe[B | None]\n        "
    bool_0 = True
    float_0 = -2409.101261
    bytes_0 = b"z\xb0\xa4f\x81\xd5\x18\xbe\x1e\x02\x97\xa7"
    float_1 = 257.687002
    tuple_0 = (float_0, bytes_0, float_1, bool_0)
    maybe_0 = module_0.Maybe(tuple_0, float_0)
    var_0 = maybe_0.bind(bool_0)
    bool_1 = var_0.__eq__(bool_0)
    var_1 = maybe_0.to_either()
    bool_2 = False
    maybe_1 = module_0.Maybe(float_0, var_1)
    maybe_2 = module_0.Maybe(str_0, bool_2)


def test_case_3():
    str_0 = ".WL"
    maybe_0 = module_0.Maybe(str_0, str_0)
    var_0 = maybe_0.map(str_0)
    var_1 = var_0.to_try()


def test_case_4():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.to_box()
    var_1 = maybe_0.bind(maybe_0)
    int_0 = 2672
    var_2 = maybe_0.ap(int_0)
    var_3 = maybe_0.filter(var_2)
    var_4 = var_2.to_box()
    var_5 = maybe_0.get_or_else(bool_0)


def test_case_5():
    str_0 = "ARB6&!~bqf^]Y@"
    bool_0 = False
    maybe_0 = module_0.Maybe(str_0, bool_0)
    maybe_0.bind(maybe_0)


def test_case_6():
    bool_0 = True
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.to_box()
    int_0 = 2672
    var_1 = maybe_0.ap(int_0)
    var_2 = maybe_0.filter(var_1)
    var_3 = var_1.to_box()
    var_4 = var_2.to_lazy()


def test_case_7():
    bool_0 = False
    bytes_0 = b"\x83/;\x02\x8d\x1a\x1b&\x1d\xb9\x1e\x822"
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.to_validation()
    var_1 = var_0.to_box()
    maybe_1 = module_0.Maybe(bytes_0, bool_0)
    maybe_1.ap(bool_0)


def test_case_8():
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.filter(none_type_0)
    maybe_1 = module_0.Maybe(none_type_0, none_type_0)
    maybe_1.filter(bool_0)


def test_case_9():
    bool_0 = False
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.get_or_else(bool_0)
    var_0.to_try()


def test_case_10():
    none_type_0 = None
    bool_0 = True
    maybe_0 = module_0.Maybe(none_type_0, bool_0)
    var_0 = maybe_0.ap(bool_0)
    var_1 = maybe_0.to_validation()
    var_2 = maybe_0.bind(var_0)
    var_3 = maybe_0.to_validation()
    maybe_1 = module_0.Maybe(none_type_0, bool_0)
    var_4 = maybe_1.to_either()
    var_5 = var_4.ap(maybe_0)


def test_case_11():
    bool_0 = False
    maybe_0 = module_0.Maybe(bool_0, bool_0)
    var_0 = maybe_0.to_try()
    var_1 = maybe_0.to_either()
    maybe_1 = module_0.Maybe(var_0, var_1)


def test_case_12():
    none_type_0 = None
    maybe_0 = module_0.Maybe(none_type_0, none_type_0)
    var_0 = maybe_0.to_box()


def test_case_13():
    bytes_0 = b"\xdc\x81\xaa\xb2\xda\xd4\xe1\x9e"
    maybe_0 = module_0.Maybe(bytes_0, bytes_0)
    var_0 = maybe_0.to_lazy()
    var_1 = maybe_0.to_try()
    maybe_1 = module_0.Maybe(bytes_0, bytes_0)
    var_2 = maybe_1.to_validation()
    var_3 = var_2.to_lazy()
    none_type_0 = None
    maybe_2 = module_0.Maybe(none_type_0, none_type_0)
    var_4 = maybe_2.to_lazy()
    var_5 = var_4.to_try()
    maybe_3 = module_0.Maybe(none_type_0, none_type_0)
    var_6 = maybe_3.to_validation()
    var_6.to_validation()


def test_case_14():
    int_0 = -1663
    bool_0 = True
    maybe_0 = module_0.Maybe(int_0, bool_0)
    var_0 = maybe_0.ap(int_0)
    var_1 = maybe_0.filter(var_0)
    var_2 = maybe_0.to_validation()
    var_3 = maybe_0.ap(var_2)
    bool_1 = False
    var_4 = maybe_0.map(bool_1)


def test_case_15():
    bytes_0 = b"\x95\xfe\xcc\xc0\xe5\xd8[\xc0~-.<\x81\x9c\xae\xec\x03>"
    none_type_0 = None
    maybe_0 = module_0.Maybe(bytes_0, none_type_0)
    var_0 = maybe_0.to_validation()
    var_0.to_validation()


def test_case_16():
    tuple_0 = ()
    maybe_0 = module_0.Maybe(tuple_0, tuple_0)
    var_0 = maybe_0.to_try()
    bool_0 = True
    maybe_1 = module_0.Maybe(tuple_0, bool_0)
    bool_1 = maybe_0.__eq__(maybe_0)
    var_1 = maybe_0.get_or_else(tuple_0)
    var_2 = maybe_0.to_try()
    var_3 = maybe_1.to_box()
    var_4 = maybe_1.ap(tuple_0)
    maybe_0.map(var_2)


def test_case_17():
    bool_0 = True
    bool_1 = False
    none_type_0 = None
    bool_2 = True
    maybe_0 = module_0.Maybe(bool_0, bool_2)
    var_0 = maybe_0.to_lazy()
    var_1 = var_0.to_box()
    var_2 = var_1.to_either()
    var_3 = var_2.to_validation()
    var_4 = var_3.to_lazy()
    var_5 = var_4.ap(bool_0)
    var_6 = var_5.map(none_type_0)
    var_6.get_or_else(bool_1)


def test_case_18():
    int_0 = 1
    str_0 = "\n        Take function (A) -> b and applied this function on current box value and returns new box with mapped value.\n\n        :param mapper: mapper function\n        :type mapper: Function(A) -> B\n        :returns: new box with mapped value\n        :rtype: Box[B]\n        "
    float_0 = 1162.0
    maybe_0 = module_0.Maybe(float_0, float_0)
    var_0 = maybe_0.bind(str_0)
    var_1 = var_0.filter(int_0)
    var_2 = var_1.to_box()
    var_3 = var_2.to_lazy()
    var_4 = var_0.ap(var_3)
    var_5 = var_4.bind(var_3)
    bool_0 = var_0.__eq__(var_4)
    var_6 = var_1.to_either()
    bool_1 = False
    maybe_1 = module_0.Maybe(var_0, bool_1)
    maybe_2 = module_0.Maybe(var_6, bool_1)


def test_case_19():
    int_0 = 1
    str_0 = "\n        Take function (A) -> b and applied this function on current box value and returns new box with mapped value.\n\n        :param mapper: mapper function\n        :type mapper: Function(A) -> B\n        :returns: new box with mapped value\n        :rtype: Box[B]\n        "
    float_0 = 1162.0
    maybe_0 = module_0.Maybe(float_0, float_0)
    var_0 = maybe_0.bind(str_0)
    var_1 = var_0.filter(int_0)
    var_2 = maybe_0.to_either()
    var_3 = var_1.to_box()
    var_4 = var_2.bind(var_1)
    bool_0 = var_0.__eq__(maybe_0)
    var_5 = var_1.to_either()
    var_6 = var_1.to_box()
    bool_1 = False
    maybe_1 = module_0.Maybe(var_0, bool_1)
    maybe_1.filter(str_0)
