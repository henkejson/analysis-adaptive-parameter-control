# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.helpers as module_0
import inspect as module_1


def test_case_0():
    bool_0 = module_0.is_atty()
    var_0 = module_0.has_message_body(bool_0)


def test_case_1():
    str_0 = "lifespan.shutdown.failed"
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0, str_0: str_0}
    var_0 = module_0.remove_entity_headers(dict_0)


def test_case_2():
    bool_0 = module_0.is_atty()


def test_case_3():
    default_0 = module_0.Default()


def test_case_4():
    none_type_0 = None
    module_0.is_hop_by_hop_header(none_type_0)


def test_case_5():
    default_0 = module_0.Default()
    str_0 = default_0.__str__()
    bool_0 = module_0.is_atty()
    var_0 = module_0.has_message_body(bool_0)
    var_1 = var_0.__repr__()
    module_0.is_hop_by_hop_header(default_0)


def test_case_6():
    float_0 = 416.95404617688615
    var_0 = module_0.has_message_body(float_0)


def test_case_7():
    str_0 = "content-location"
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0, str_0: str_0, str_0: str_0}
    var_0 = module_0.remove_entity_headers(dict_0, str_0)
    bool_0 = module_0.is_atty()
    var_1 = module_0.has_message_body(bool_0)
    default_0 = module_0.Default()
    var_2 = module_1.ismodule(var_1)
    int_0 = 191
    var_3 = module_0.has_message_body(int_0)
    str_1 = str_0.__str__()
    str_2 = var_1.__str__()
    str_3 = default_0.__str__()


def test_case_8():
    str_0 = "content-location"
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0, str_0: str_0, str_0: str_0}
    var_0 = module_0.remove_entity_headers(dict_0)
    bool_0 = module_0.is_atty()
    module_0.has_message_body(str_0)


def test_case_9():
    str_0 = "content-location"
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0, str_0: str_0, str_0: str_0}
    var_0 = module_0.remove_entity_headers(dict_0, str_0)
    bool_0 = module_0.is_atty()
    var_1 = module_0.has_message_body(bool_0)
    default_0 = module_0.Default()
    var_2 = module_1.ismodule(var_1)
    var_3 = default_0.__repr__()
    str_1 = var_1.__str__()
    float_0 = 304.0
    var_4 = module_0.has_message_body(float_0)
    module_0.has_message_body(var_3)
