# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.helpers as module_0
import inspect as module_1


def test_case_0():
    int_0 = 164
    var_0 = module_0.has_message_body(int_0)


def test_case_1():
    bool_0 = True
    var_0 = module_0.has_message_body(bool_0)
    module_0.remove_entity_headers(bool_0)


def test_case_2():
    bool_0 = module_0.is_atty()


def test_case_3():
    str_0 = "#Rqz6\x0bz(K#{"
    dict_0 = {str_0: str_0}
    var_0 = module_0.remove_entity_headers(dict_0)


def test_case_4():
    complex_0 = -2079.9 - 528j
    str_0 = "Zs$7j9%\n5 H\t9sk8K"
    tuple_0 = (complex_0, complex_0, complex_0, str_0)
    module_0.is_hop_by_hop_header(tuple_0)


def test_case_5():
    default_0 = module_0.Default()
    str_0 = default_0.__str__()
    default_1 = module_0.Default()


def test_case_6():
    default_0 = module_0.Default()
    str_0 = default_0.__str__()
    var_0 = default_0.__repr__()
    str_1 = default_0.__str__()
    default_1 = module_0.Default()
    var_1 = default_0.__repr__()
    str_2 = default_1.__str__()
    var_2 = var_1.__repr__()
    var_3 = var_1.__repr__()
    bool_0 = module_0.is_atty()
    var_4 = module_1.ismodule(bool_0)
    str_3 = var_3.__str__()
    dict_0 = {var_2: var_4, str_3: default_0, var_3: var_2}
    var_5 = module_0.remove_entity_headers(dict_0)
    var_6 = module_0.is_hop_by_hop_header(str_3)
    var_7 = module_1.ismodule(var_6)
    var_8 = default_0.__repr__()
    var_9 = var_3.__repr__()
    default_2 = module_0.Default()
    var_10 = module_1.ismodule(var_2)
    int_0 = 204
    var_11 = module_0.has_message_body(int_0)
    var_12 = var_7.__repr__()
    module_0.has_message_body(var_8)
