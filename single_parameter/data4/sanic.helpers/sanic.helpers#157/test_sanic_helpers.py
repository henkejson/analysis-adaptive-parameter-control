# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.helpers as module_0
import inspect as module_1


def test_case_0():
    bool_0 = module_0.is_atty()
    var_0 = module_0.has_message_body(bool_0)


def test_case_1():
    bool_0 = False
    dict_0 = {bool_0: bool_0, bool_0: bool_0, bool_0: bool_0, bool_0: bool_0}
    module_0.remove_entity_headers(dict_0)


def test_case_2():
    bool_0 = module_0.is_atty()


def test_case_3():
    list_0 = []
    module_0.is_hop_by_hop_header(list_0)


def test_case_4():
    default_0 = module_0.Default()
    var_0 = default_0.__repr__()
    dict_0 = {var_0: var_0, var_0: var_0, var_0: var_0, var_0: var_0}
    var_1 = module_0.remove_entity_headers(dict_0)


def test_case_5():
    default_0 = module_0.Default()
    str_0 = default_0.__str__()
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0, str_0: str_0}
    var_0 = module_0.remove_entity_headers(dict_0)


def test_case_6():
    bool_0 = module_0.is_atty()
    int_0 = 203
    var_0 = module_0.has_message_body(int_0)


def test_case_7():
    default_0 = module_0.Default()
    var_0 = default_0.__repr__()
    str_0 = default_0.__str__()
    var_1 = module_0.is_hop_by_hop_header(var_0)
    dict_0 = {var_0: var_0, var_0: var_0, var_0: var_0, var_0: var_0}
    var_2 = module_0.remove_entity_headers(dict_0)
    var_3 = default_0.__repr__()
    var_4 = module_1.ismodule(default_0)
    var_5 = module_1.ismodule(var_0)
    bool_0 = module_0.is_atty()
    var_6 = module_0.has_message_body(var_1)
    str_1 = var_4.__str__()
    var_7 = var_4.__repr__()
    int_0 = 204
    var_8 = module_0.has_message_body(int_0)
    var_9 = module_0.is_hop_by_hop_header(str_0)
    var_10 = var_7.__repr__()
