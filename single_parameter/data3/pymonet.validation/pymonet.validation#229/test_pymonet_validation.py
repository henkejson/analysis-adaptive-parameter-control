# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.validation as module_0


def test_case_0():
    bool_0 = True
    validation_0 = module_0.Validation(bool_0, bool_0)
    bool_1 = False
    validation_1 = module_0.Validation(bool_1, bool_1)
    var_0 = validation_1.to_box()
    var_1 = var_0.to_either()
    var_2 = validation_0.__eq__(var_1)
    var_1.ap(bool_0)


def test_case_1():
    str_0 = "Brs#k=TOKY[N^=3)bO;-"
    list_0 = [str_0, str_0]
    validation_0 = module_0.Validation(list_0, list_0)
    var_0 = validation_0.__str__()
    tuple_0 = (str_0, list_0)
    tuple_0.to_maybe()


def test_case_2():
    none_type_0 = None
    none_type_0.to_try()


def test_case_3():
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, none_type_0)


def test_case_4():
    bool_0 = False
    validation_0 = module_0.Validation(bool_0, bool_0)
    validation_0.to_try()


def test_case_5():
    bool_0 = False
    validation_0 = module_0.Validation(bool_0, bool_0)
    validation_1 = module_0.Validation(validation_0, bool_0)
    var_0 = validation_1.__eq__(validation_1)
    validation_1.is_fail()


def test_case_6():
    none_type_0 = None
    none_type_1 = None
    validation_0 = module_0.Validation(none_type_1, none_type_1)
    validation_0.map(none_type_0)


def test_case_7():
    bytes_0 = b">\x13v\x9fR\xdb\x8e\x9e\x03I\x0c\xaf\xc1"
    dict_0 = {bytes_0: bytes_0}
    none_type_0 = None
    validation_0 = module_0.Validation(none_type_0, dict_0)
    validation_0.bind(dict_0)


def test_case_8():
    bool_0 = False
    validation_0 = module_0.Validation(bool_0, bool_0)
    var_0 = validation_0.__eq__(validation_0)
    validation_0.ap(validation_0)


def test_case_9():
    bool_0 = False
    validation_0 = module_0.Validation(bool_0, bool_0)
    var_0 = validation_0.to_lazy()
    var_1 = var_0.to_try()
    var_1.map(bool_0)


def test_case_10():
    bool_0 = False
    validation_0 = module_0.Validation(bool_0, bool_0)
    var_0 = validation_0.__eq__(validation_0)
    bool_0.map(bool_0)


def test_case_11():
    bytes_0 = b"\x0e\xa5\xb0*\x1em\xb4\xe8\xc7V\xd6\x92|A"
    validation_0 = module_0.Validation(bytes_0, bytes_0)
    var_0 = validation_0.to_box()
    var_1 = validation_0.__eq__(bytes_0)
    var_2 = validation_0.to_maybe()
    validation_0.ap(var_2)


def test_case_12():
    bytes_0 = b""
    validation_0 = module_0.Validation(bytes_0, bytes_0)
    var_0 = validation_0.to_box()
    var_1 = validation_0.__eq__(bytes_0)
    var_2 = validation_0.to_either()
    var_3 = validation_0.to_maybe()
    var_3.ap(bytes_0)


def test_case_13():
    bytes_0 = b""
    validation_0 = module_0.Validation(bytes_0, bytes_0)
    var_0 = validation_0.to_box()
    var_1 = validation_0.__eq__(bytes_0)
    var_2 = validation_0.to_maybe()
    var_2.ap(bytes_0)


def test_case_14():
    bytes_0 = b',\xc3\xfa\x97S\x98O\x1d\x07\x90\x01\xb0"D\rH(uR'
    validation_0 = module_0.Validation(bytes_0, bytes_0)
    var_0 = validation_0.to_box()
    var_1 = var_0.to_maybe()
    var_2 = validation_0.to_either()
    var_3 = var_1.to_box()
    var_3.ap(bytes_0)


def test_case_15():
    bytes_0 = b"\xd08T\xcf\xa3\xe4\x12\x05\x8d~Pm\xd7"
    none_type_0 = None
    dict_0 = {}
    validation_0 = module_0.Validation(none_type_0, dict_0)
    var_0 = validation_0.__str__()
    var_0.map(bytes_0)


def test_case_16():
    bool_0 = False
    validation_0 = module_0.Validation(bool_0, bool_0)
    validation_1 = module_0.Validation(validation_0, validation_0)
    var_0 = validation_0.__eq__(validation_1)
    validation_1.to_try()
