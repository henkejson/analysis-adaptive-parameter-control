# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.helpers as module_0
import inspect as module_1
import builtins as module_2


def test_case_0():
    bytes_0 = b"Forbidden"
    module_0.has_message_body(bytes_0)


def test_case_1():
    bytes_0 = b"\xc2"
    dict_0 = {bytes_0: bytes_0, bytes_0: bytes_0}
    var_0 = module_0.remove_entity_headers(dict_0)


def test_case_2():
    bool_0 = module_0.is_atty()


def test_case_3():
    default_0 = module_0.Default()


def test_case_4():
    bytes_0 = b"Forbidden"
    var_0 = module_0.is_entity_header(bytes_0)


def test_case_5():
    bool_0 = False
    module_0.is_hop_by_hop_header(bool_0)


def test_case_6():
    set_0 = set()
    default_0 = module_0.Default()
    var_0 = default_0.__repr__()
    module_0.remove_entity_headers(set_0)


def test_case_7():
    default_0 = module_0.Default()
    default_1 = module_0.Default()
    str_0 = default_0.__str__()
    module_0.has_message_body(default_0)


def test_case_8():
    bytes_0 = b"\x1c\x0fOz\x0bU\xaf\xc6_y"
    int_0 = 1343
    var_0 = module_0.has_message_body(int_0)
    str_0 = var_0.__str__()
    bool_0 = False
    dict_0 = {bytes_0: bool_0}
    var_1 = module_0.remove_entity_headers(dict_0)
    str_1 = var_1.__str__()
    var_2 = module_0.has_message_body(var_0)
    none_type_0 = None
    module_0.has_message_body(none_type_0)


def test_case_9():
    bool_0 = True
    var_0 = module_0.has_message_body(bool_0)


def test_case_10():
    bytes_0 = b"\x1c\x0fOz\x0bU\xaf\xc6_y"
    int_0 = 304
    var_0 = module_0.has_message_body(int_0)
    var_1 = module_0.has_message_body(var_0)
    str_0 = var_0.__str__()
    bool_0 = False
    dict_0 = {bytes_0: bool_0}
    var_2 = module_0.remove_entity_headers(dict_0)
    var_3 = var_2.__repr__()
    bool_1 = module_0.is_atty()
    var_4 = module_1.ismodule(var_0)
    var_5 = var_4.__repr__()
    object_0 = module_2.object()
    bytes_1 = b",\x9e_K"
    module_0.has_message_body(bytes_1)
