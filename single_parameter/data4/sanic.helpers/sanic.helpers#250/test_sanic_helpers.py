# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.helpers as module_0
import inspect as module_1


def test_case_0():
    dict_0 = {}
    module_0.has_message_body(dict_0)


def test_case_1():
    bool_0 = module_0.is_atty()
    module_0.remove_entity_headers(bool_0)


def test_case_2():
    bool_0 = module_0.is_atty()


def test_case_3():
    default_0 = module_0.Default()
    str_0 = default_0.__str__()
    dict_0 = {str_0: default_0}
    var_0 = module_0.remove_entity_headers(dict_0)


def test_case_4():
    int_0 = 2207
    module_0.is_hop_by_hop_header(int_0)


def test_case_5():
    none_type_0 = None
    module_0.remove_entity_headers(none_type_0, none_type_0)


def test_case_6():
    default_0 = module_0.Default()
    str_0 = default_0.__str__()
    int_0 = 310
    var_0 = module_0.has_message_body(int_0)
    dict_0 = {str_0: var_0}
    var_1 = module_0.remove_entity_headers(dict_0)


def test_case_7():
    bool_0 = True
    var_0 = module_0.has_message_body(bool_0)
    dict_0 = {}
    module_0.has_message_body(dict_0)


def test_case_8():
    default_0 = module_0.Default()
    str_0 = default_0.__str__()
    int_0 = 304
    var_0 = module_0.has_message_body(int_0)
    bool_0 = module_0.is_atty()
    var_1 = default_0.__repr__()
    dict_0 = {str_0: var_0}
    var_2 = module_0.has_message_body(bool_0)
    var_3 = module_0.remove_entity_headers(dict_0)
    var_4 = module_1.ismodule(dict_0)
    var_5 = module_0.has_message_body(bool_0)
    module_0.remove_entity_headers(int_0)
