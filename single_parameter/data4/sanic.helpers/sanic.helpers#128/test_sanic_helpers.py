# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.helpers as module_0
import inspect as module_1


def test_case_0():
    int_0 = 8443
    var_0 = module_0.has_message_body(int_0)
    var_1 = module_1.ismodule(int_0)


def test_case_1():
    int_0 = 8443
    var_0 = module_1.ismodule(int_0)
    module_0.remove_entity_headers(var_0)


def test_case_2():
    bool_0 = module_0.is_atty()


def test_case_3():
    bool_0 = module_0.is_atty()
    module_0.is_entity_header(bool_0)


def test_case_4():
    bool_0 = True
    module_0.is_hop_by_hop_header(bool_0)


def test_case_5():
    bool_0 = module_0.is_atty()
    default_0 = module_0.Default()
    var_0 = default_0.__repr__()


def test_case_6():
    bool_0 = module_0.is_atty()
    bool_1 = module_0.is_atty()
    default_0 = module_0.Default()
    int_0 = 229
    str_0 = default_0.__str__()
    var_0 = module_0.has_message_body(int_0)
    str_1 = ""
    var_1 = var_0.__repr__()
    var_2 = module_0.is_hop_by_hop_header(str_1)
    module_0.import_string(str_1)


def test_case_7():
    bool_0 = module_0.is_atty()
    int_0 = 58
    var_0 = module_0.has_message_body(int_0)


def test_case_8():
    str_0 = "The client accepts "
    dict_0 = {str_0: str_0, str_0: str_0}
    var_0 = module_0.remove_entity_headers(dict_0)
    module_0.has_message_body(var_0)


def test_case_9():
    str_0 = "The clEient accepts "
    dict_0 = {str_0: str_0, str_0: str_0}
    var_0 = module_0.remove_entity_headers(dict_0)
    bool_0 = module_0.is_atty()
    int_0 = 204
    var_1 = module_0.has_message_body(int_0)
    str_1 = "1"
    var_2 = module_0.is_hop_by_hop_header(str_1)
    module_0.remove_entity_headers(bool_0)


def test_case_10():
    str_0 = "Content-Length"
    dict_0 = {str_0: str_0, str_0: str_0}
    var_0 = module_0.remove_entity_headers(dict_0)
    str_1 = "1"
    var_1 = module_0.is_hop_by_hop_header(str_1)
    module_0.remove_entity_headers(str_1)


def test_case_11():
    str_0 = "Content-Length"
    dict_0 = {str_0: str_0, str_0: str_0}
    var_0 = module_0.remove_entity_headers(dict_0, dict_0)
    var_1 = module_0.remove_entity_headers(dict_0)
    bool_0 = module_0.is_atty()
    int_0 = 203
    var_2 = module_0.has_message_body(int_0)
    str_1 = "1"
    var_3 = module_0.is_hop_by_hop_header(str_1)
    module_0.import_string(bool_0)
