# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.helpers as module_0


def test_case_0():
    int_0 = 2588
    var_0 = module_0.has_message_body(int_0)
    var_1 = var_0.__repr__()
    var_2 = var_1.__repr__()


def test_case_1():
    dict_0 = {}
    var_0 = module_0.remove_entity_headers(dict_0)


def test_case_2():
    dict_0 = {}
    default_0 = module_0.Default(**dict_0)
    var_0 = default_0.__repr__()
    dict_1 = {var_0: var_0, var_0: var_0}
    var_1 = module_0.remove_entity_headers(dict_1)


def test_case_3():
    bool_0 = module_0.is_atty()


def test_case_4():
    none_type_0 = None
    module_0.is_entity_header(none_type_0)


def test_case_5():
    list_0 = []
    bool_0 = module_0.is_atty()
    module_0.is_hop_by_hop_header(list_0)


def test_case_6():
    bool_0 = module_0.is_atty()
    default_0 = module_0.Default()
    bool_1 = default_0.__repr__()
    str_0 = default_0.__str__()
    bool_2 = module_0.is_atty()


def test_case_7():
    bool_0 = False
    var_0 = module_0.has_message_body(bool_0)
    str_0 = var_0.__str__()
    bool_1 = module_0.is_atty()


def test_case_8():
    dict_0 = {}
    bool_0 = module_0.is_atty()
    default_0 = module_0.Default(**dict_0)
    var_0 = module_0.remove_entity_headers(dict_0)
    str_0 = default_0.__str__()
    none_type_0 = None
    int_0 = 204
    var_1 = module_0.has_message_body(int_0)
    var_2 = module_0.has_message_body(bool_0)
    var_3 = module_0.has_message_body(bool_0)
    module_0.is_entity_header(none_type_0)
