# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.helpers as module_0
import builtins as module_1
import inspect as module_2


def test_case_0():
    list_0 = []
    module_0.has_message_body(list_0)


def test_case_1():
    float_0 = -2115.0
    dict_0 = {float_0: float_0, float_0: float_0, float_0: float_0}
    module_0.remove_entity_headers(dict_0)


def test_case_2():
    bool_0 = module_0.is_atty()


def test_case_3():
    str_0 = "'None' was returned while requesting a handler from the router"
    dict_0 = {str_0: str_0, str_0: str_0}
    var_0 = module_0.remove_entity_headers(dict_0)
    module_0.is_hop_by_hop_header(dict_0)


def test_case_4():
    default_0 = module_0.Default()
    var_0 = default_0.__repr__()
    str_0 = default_0.__str__()


def test_case_5():
    float_0 = -2115.0
    var_0 = module_0.has_message_body(float_0)
    dict_0 = {float_0: float_0, float_0: float_0, float_0: float_0}
    var_1 = var_0.__repr__()
    module_0.remove_entity_headers(dict_0)


def test_case_6():
    float_0 = 339.0
    var_0 = module_0.has_message_body(float_0)
    object_0 = module_1.object()
    var_1 = var_0.__repr__()
    bool_0 = module_0.is_atty()
    default_0 = module_0.Default()
    module_0.is_hop_by_hop_header(var_0)


def test_case_7():
    str_0 = "'None' was returned while requesting a handler from the router"
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0}
    var_0 = module_0.remove_entity_headers(dict_0)


def test_case_8():
    str_0 = "Z|G@Fg."
    default_0 = module_0.Default()
    str_1 = default_0.__str__()
    dict_0 = {str_0: default_0, str_0: default_0}
    var_0 = module_0.remove_entity_headers(dict_0)
    str_2 = default_0.__str__()
    default_1 = module_0.Default()
    float_0 = 304.0
    bool_0 = module_0.is_atty()
    var_1 = module_0.has_message_body(float_0)
    default_2 = module_0.Default()
    var_2 = module_2.ismodule(str_2)


def test_case_9():
    str_0 = "extension-header"
    default_0 = module_0.Default()
    var_0 = module_0.is_hop_by_hop_header(str_0)
    dict_0 = {str_0: default_0, str_0: default_0}
    var_1 = module_0.remove_entity_headers(dict_0)
    bool_0 = module_0.is_atty()
    var_2 = module_0.has_message_body(bool_0)
    default_1 = module_0.Default()


def test_case_10():
    str_0 = "extension-header"
    default_0 = module_0.Default()
    var_0 = module_0.is_hop_by_hop_header(str_0)
    dict_0 = {str_0: default_0, str_0: default_0}
    var_1 = module_0.remove_entity_headers(dict_0)
    str_1 = default_0.__str__()
    default_1 = module_0.Default()
    float_0 = 303.6322361829123
    bool_0 = module_2.ismodule(var_1)
    var_2 = module_0.has_message_body(float_0)
    default_2 = module_0.Default()
    default_3 = module_0.remove_entity_headers(dict_0, dict_0)
    bytes_0 = b";\xd2\x927\x97\xbb\xf0J\x95\x98\xd95:H\x8emq\x8d\xda"
    var_3 = module_0.is_hop_by_hop_header(bytes_0)
