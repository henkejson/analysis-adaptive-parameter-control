# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.helpers as module_0
import inspect as module_1


def test_case_0():
    bool_0 = True
    var_0 = module_0.has_message_body(bool_0)


def test_case_1():
    str_0 = "hK\x0c:3(? HGd\n6pz6d"
    module_0.remove_entity_headers(str_0)


def test_case_2():
    bool_0 = module_0.is_atty()


def test_case_3():
    bool_0 = False
    var_0 = module_0.has_message_body(bool_0)
    var_1 = var_0.__repr__()
    module_0.is_entity_header(bool_0)


def test_case_4():
    str_0 = "l9wza"
    list_0 = [str_0]
    module_0.is_hop_by_hop_header(list_0)


def test_case_5():
    int_0 = 305
    var_0 = module_0.has_message_body(int_0)
    dict_0 = {}
    var_1 = module_0.remove_entity_headers(dict_0)
    bool_0 = module_0.is_atty()
    default_0 = module_0.Default()
    str_0 = default_0.__str__()
    module_0.is_hop_by_hop_header(bool_0)


def test_case_6():
    bool_0 = module_0.is_atty()
    var_0 = bool_0.__repr__()
    dict_0 = {var_0: var_0, var_0: var_0}
    var_1 = module_0.remove_entity_headers(dict_0)


def test_case_7():
    int_0 = 135
    var_0 = module_0.has_message_body(int_0)


def test_case_8():
    int_0 = 304
    var_0 = module_1.ismodule(int_0)
    str_0 = var_0.__str__()
    var_1 = module_0.has_message_body(int_0)
    dict_0 = {}
    var_2 = module_0.remove_entity_headers(dict_0)
    str_1 = "VVi)+gp"
    var_3 = module_0.is_hop_by_hop_header(str_1)
    var_4 = var_2.__repr__()
    str_2 = var_4.__str__()
    default_0 = module_0.Default()
    module_0.has_message_body(var_2)
