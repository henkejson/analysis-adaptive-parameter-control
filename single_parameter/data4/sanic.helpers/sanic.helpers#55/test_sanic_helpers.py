# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.helpers as module_0


def test_case_0():
    bool_0 = module_0.is_atty()
    var_0 = module_0.has_message_body(bool_0)


def test_case_1():
    bytes_0 = b"\x82*\xfe\xfa\x7fu?{ \x06\x14\x95\xcf)\x9fhs"
    dict_0 = {bytes_0: bytes_0, bytes_0: bytes_0}
    var_0 = module_0.remove_entity_headers(dict_0)


def test_case_2():
    bool_0 = module_0.is_atty()


def test_case_3():
    none_type_0 = None
    module_0.is_entity_header(none_type_0)


def test_case_4():
    bytes_0 = b"\x82*\xfe\xfa\x7fu?{ \x06\x14\x95\xcf)\x9fhs"
    dict_0 = {bytes_0: bytes_0, bytes_0: bytes_0}
    var_0 = module_0.remove_entity_headers(dict_0)
    str_0 = module_0.is_hop_by_hop_header(bytes_0)


def test_case_5():
    default_0 = module_0.Default()
    str_0 = "It seems that one or more of your workers failed to come online in the allowed time. Sanic is shutting down to avoid a deadlock. The current threshold is "
    str_1 = default_0.__str__()
    module_0.import_string(str_0)


def test_case_6():
    int_0 = 1079
    var_0 = module_0.has_message_body(int_0)
    bool_0 = module_0.has_message_body(var_0)
