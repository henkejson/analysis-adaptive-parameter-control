# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.helpers as module_0


def test_case_0():
    str_0 = "Eur~;x~jfTdOaq?.=fQh"
    module_0.has_message_body(str_0)


def test_case_1():
    bytes_0 = b"\x08\xd3\xc8\x80\x95\xb0/\xca\xfd\xfe+mW'"
    dict_0 = {bytes_0: bytes_0, bytes_0: bytes_0, bytes_0: bytes_0}
    var_0 = module_0.remove_entity_headers(dict_0)


def test_case_2():
    bool_0 = module_0.is_atty()


def test_case_3():
    none_type_0 = None
    module_0.is_entity_header(none_type_0)


def test_case_4():
    default_0 = module_0.Default()
    str_0 = default_0.__str__()
    int_0 = 205
    var_0 = module_0.has_message_body(int_0)
    var_1 = module_0.is_hop_by_hop_header(str_0)


def test_case_5():
    bool_0 = False
    module_0.remove_entity_headers(bool_0, bool_0)


def test_case_6():
    default_0 = module_0.Default()
    str_0 = default_0.__str__()
    int_0 = 205
    var_0 = module_0.has_message_body(int_0)


def test_case_7():
    bool_0 = module_0.is_atty()
    var_0 = module_0.has_message_body(bool_0)
    var_1 = module_0.has_message_body(var_0)


def test_case_8():
    int_0 = 205
    var_0 = module_0.has_message_body(int_0)


def test_case_9():
    bool_0 = module_0.is_atty()
    default_0 = module_0.Default()
    str_0 = default_0.__str__()
    var_0 = module_0.has_message_body(bool_0)
    var_1 = var_0.__repr__()
    dict_0 = {var_1: bool_0, var_1: var_1}
    var_2 = module_0.remove_entity_headers(dict_0)
    int_0 = 204
    var_3 = module_0.has_message_body(int_0)
    var_4 = var_2.__repr__()
    str_1 = var_1.__str__()
    var_5 = module_0.is_hop_by_hop_header(str_0)
    var_6 = module_0.is_entity_header(var_1)
    var_7 = module_0.is_hop_by_hop_header(var_1)
    module_0.import_string(var_3, var_1)
