# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.helpers as module_0
import builtins as module_1


def test_case_0():
    none_type_0 = None
    module_0.has_message_body(none_type_0)


def test_case_1():
    list_0 = []
    module_0.remove_entity_headers(list_0)


def test_case_2():
    bool_0 = module_0.is_atty()


def test_case_3():
    bool_0 = module_0.is_atty()
    var_0 = bool_0.__repr__()
    dict_0 = {var_0: var_0, bool_0: var_0, var_0: bool_0, var_0: var_0}
    module_0.remove_entity_headers(dict_0)


def test_case_4():
    bytes_0 = b"\xe9r\xedt\xc2\xb9\x10\xed\xb5\xbf\x1eJ+U?=\xce"
    var_0 = module_0.is_hop_by_hop_header(bytes_0)


def test_case_5():
    int_0 = 429
    default_0 = module_0.Default()
    var_0 = default_0.__repr__()
    var_1 = module_0.has_message_body(int_0)
    var_2 = var_1.__repr__()
    module_0.Default(**var_2)


def test_case_6():
    int_0 = 158
    var_0 = module_0.has_message_body(int_0)
    bool_0 = module_0.is_atty()
    var_1 = module_0.has_message_body(bool_0)
    bytes_0 = b"No Content"
    var_2 = module_0.is_hop_by_hop_header(bytes_0)
    var_3 = var_0.__repr__()
    bool_1 = True
    default_0 = module_0.Default()
    str_0 = default_0.__str__()
    var_4 = bool_0.__repr__()
    dict_0 = {var_4: var_4, bool_0: var_4, bool_1: bool_0, bool_0: var_4}
    str_1 = bool_0.__str__()
    module_0.remove_entity_headers(dict_0)


def test_case_7():
    bool_0 = module_0.is_atty()
    var_0 = module_0.has_message_body(bool_0)


def test_case_8():
    bool_0 = False
    int_0 = 100
    var_0 = module_0.has_message_body(int_0)
    var_1 = bool_0.__repr__()


def test_case_9():
    int_0 = 147
    var_0 = module_0.has_message_body(int_0)
    var_1 = module_0.has_message_body(var_0)
    var_2 = var_0.__repr__()
    dict_0 = {var_2: var_2, var_2: var_2, var_2: var_2, var_2: var_2}
    str_0 = var_0.__str__()
    var_3 = module_0.remove_entity_headers(dict_0)
    module_0.import_string(dict_0, dict_0)


def test_case_10():
    int_0 = 204
    default_0 = module_0.Default()
    var_0 = module_0.has_message_body(int_0)
    var_1 = default_0.__repr__()
    bool_0 = module_0.is_atty()
    var_2 = default_0.__repr__()
    bytes_0 = b"\x04N*\xa6o \xd2on\xa5et"
    object_0 = module_1.object()
    var_3 = module_0.is_hop_by_hop_header(bytes_0)
    var_4 = module_0.has_message_body(var_0)
    bool_1 = False
    bool_2 = module_0.is_atty()
    var_5 = bool_0.__repr__()
    dict_0 = {var_5: var_5, bool_0: var_5, bool_1: bool_0, bool_0: var_5}
    str_0 = default_0.__str__()
    module_0.remove_entity_headers(dict_0)
