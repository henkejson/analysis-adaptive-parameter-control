# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.helpers as module_0
import builtins as module_1
import inspect as module_2


def test_case_0():
    bool_0 = module_0.is_atty()
    int_0 = 64
    var_0 = module_0.has_message_body(int_0)


def test_case_1():
    str_0 = "Which HTTP version to use: HTTP/1.1 or HTTP/3. Value should\nbe either 1, or 3. [default 1]"
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
    object_0 = module_1.object()
    bool_0 = module_0.is_atty()
    int_0 = 78
    str_0 = default_0.__str__()
    var_0 = module_0.has_message_body(int_0)


def test_case_6():
    int_0 = 114
    var_0 = module_0.has_message_body(int_0)
    int_1 = -396
    none_type_0 = None
    module_0.remove_entity_headers(int_1, none_type_0)


def test_case_7():
    str_0 = "Which HTTP version to use: HTTP/1.1 oG HTTP/3. Value shoulobe either 1,]or 3 [default 1]"
    var_0 = module_0.is_hop_by_hop_header(str_0)
    bool_0 = module_0.is_atty()
    default_0 = module_0.Default()
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0, str_0: str_0}
    str_1 = default_0.__str__()
    var_1 = module_0.remove_entity_headers(dict_0)
    none_type_0 = None
    object_0 = module_1.object()
    var_2 = module_2.ismodule(dict_0)
    int_0 = 304
    str_2 = default_0.__str__()
    var_3 = module_0.has_message_body(int_0)
    var_4 = module_0.has_message_body(var_3)
    str_3 = var_2.__str__()
    bool_1 = module_0.is_atty()
    module_0.is_hop_by_hop_header(none_type_0)
