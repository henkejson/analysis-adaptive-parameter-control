# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.helpers as module_0
import inspect as module_1


def test_case_0():
    int_0 = 200
    var_0 = module_0.has_message_body(int_0)


def test_case_1():
    str_0 = "Wp14\\%~QTZ_P^yX5;p"
    dict_0 = {str_0: str_0}
    var_0 = module_0.remove_entity_headers(dict_0)


def test_case_2():
    bool_0 = module_0.is_atty()


def test_case_3():
    default_0 = module_0.Default()


def test_case_4():
    set_0 = set()
    int_0 = 3011
    list_0 = [set_0, set_0, set_0, int_0]
    module_0.is_hop_by_hop_header(list_0)


def test_case_5():
    int_0 = 8192
    default_0 = module_0.Default()
    str_0 = default_0.__str__()
    module_0.remove_entity_headers(int_0)


def test_case_6():
    str_0 = "Z"
    dict_0 = {str_0: str_0}
    var_0 = module_0.remove_entity_headers(dict_0)
    float_0 = -2323.5
    var_1 = module_0.has_message_body(float_0)


def test_case_7():
    str_0 = "F\tP\x0b)\tU?'Q_"
    dict_0 = {str_0: str_0}
    var_0 = module_0.remove_entity_headers(dict_0)
    var_1 = var_0.__repr__()
    float_0 = -781.8654876655694
    var_2 = var_0.__repr__()
    int_0 = 203
    var_3 = module_0.has_message_body(int_0)
    int_1 = 204
    var_4 = module_0.has_message_body(int_1)
    bool_0 = module_0.is_atty()
    str_1 = var_3.__str__()
    module_0.is_hop_by_hop_header(float_0)


def test_case_8():
    str_0 = "extension-header"
    dict_0 = {str_0: str_0}
    var_0 = module_0.remove_entity_headers(dict_0)
    var_1 = module_0.remove_entity_headers(var_0)
    module_0.is_hop_by_hop_header(var_1)


def test_case_9():
    str_0 = "extension-header"
    dict_0 = {str_0: str_0, str_0: str_0}
    var_0 = module_0.remove_entity_headers(dict_0)
    bool_0 = False
    float_0 = -780.8366
    var_1 = module_1.ismodule(var_0)
    var_2 = module_0.remove_entity_headers(dict_0, dict_0)
    var_3 = module_0.has_message_body(float_0)
    var_4 = module_0.has_message_body(float_0)
    dict_1 = {float_0: dict_0}
    bool_1 = module_0.is_atty()
    list_0 = [dict_1, bool_0, str_0]
    module_0.is_hop_by_hop_header(list_0)
