# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.helpers as module_0
import inspect as module_1


def test_case_0():
    list_0 = []
    module_0.has_message_body(list_0)


def test_case_1():
    bytes_0 = b"\xf9\xd1\xfa\x07\xf1\x08P\xe5\xdf\xda\xe5k\xd4A\xeb\xf0\xeb"
    dict_0 = {bytes_0: bytes_0, bytes_0: bytes_0}
    var_0 = module_0.remove_entity_headers(dict_0)


def test_case_2():
    bool_0 = module_0.is_atty()


def test_case_3():
    bytes_0 = b"%b?%b"
    tuple_0 = (bytes_0,)
    module_0.is_hop_by_hop_header(tuple_0)


def test_case_4():
    bool_0 = module_0.is_atty()
    int_0 = 5
    var_0 = module_1.ismodule(int_0)
    bool_1 = module_0.is_atty()
    var_1 = module_0.has_message_body(int_0)
    str_0 = var_1.__str__()
    var_2 = var_1.__repr__()
    bool_2 = module_0.is_atty()
    default_0 = module_0.Default()
    var_3 = var_1.__repr__()
    str_1 = default_0.__str__()


def test_case_5():
    float_0 = 353.4006
    var_0 = module_0.has_message_body(float_0)
    var_1 = var_0.__repr__()
    var_2 = module_1.ismodule(var_0)
    str_0 = var_2.__str__()


def test_case_6():
    float_0 = 353.4006
    var_0 = module_0.has_message_body(float_0)
    var_1 = var_0.__repr__()
    var_2 = module_0.has_message_body(var_0)
    var_3 = module_1.ismodule(var_0)
    str_0 = var_3.__str__()


def test_case_7():
    bytes_0 = b"Method Not Allowed"
    dict_0 = {bytes_0: bytes_0}
    var_0 = module_0.remove_entity_headers(dict_0)
    bool_0 = module_0.is_atty()
    int_0 = 204
    var_1 = module_0.has_message_body(int_0)
    var_2 = var_1.__repr__()
    var_3 = module_1.ismodule(int_0)
    var_4 = module_0.has_message_body(var_1)
    str_0 = var_4.__str__()
    module_0.is_hop_by_hop_header(var_0)
