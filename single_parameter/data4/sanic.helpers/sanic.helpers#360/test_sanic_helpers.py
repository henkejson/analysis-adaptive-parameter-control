# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.helpers as module_0
import inspect as module_1


def test_case_0():
    float_0 = 545.7
    var_0 = module_0.has_message_body(float_0)
    str_0 = var_0.__str__()
    module_0.is_entity_header(float_0)


def test_case_1():
    set_0 = set()
    module_0.remove_entity_headers(set_0)


def test_case_2():
    bool_0 = module_0.is_atty()


def test_case_3():
    bool_0 = module_0.is_atty()
    module_0.is_entity_header(bool_0)


def test_case_4():
    bool_0 = True
    module_0.is_hop_by_hop_header(bool_0)


def test_case_5():
    int_0 = -1631
    module_0.remove_entity_headers(int_0, int_0)


def test_case_6():
    default_0 = module_0.Default()
    dict_0 = {}
    var_0 = module_0.remove_entity_headers(dict_0, dict_0)
    default_1 = module_0.Default()
    var_1 = var_0.__repr__()
    str_0 = default_1.__str__()
    str_1 = var_1.__str__()


def test_case_7():
    float_0 = 545.7
    var_0 = module_0.has_message_body(float_0)
    str_0 = module_0.has_message_body(var_0)
    module_0.is_entity_header(float_0)


def test_case_8():
    int_0 = 204
    var_0 = module_0.has_message_body(int_0)
    bool_0 = module_0.is_atty()
    var_1 = module_0.has_message_body(var_0)
    var_2 = var_0.__repr__()
    default_0 = module_0.Default()
    str_0 = module_1.ismodule(default_0)
    var_3 = module_0.is_hop_by_hop_header(var_2)
    module_0.remove_entity_headers(default_0)


def test_case_9():
    str_0 = "content-md5"
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0}
    var_0 = module_0.remove_entity_headers(dict_0)
    str_1 = var_0.__str__()


def test_case_10():
    str_0 = "`\t<>I"
    dict_0 = {str_0: str_0}
    var_0 = module_0.remove_entity_headers(dict_0, dict_0)
    var_1 = var_0.__repr__()
    dict_1 = {str_0: str_0, str_0: str_0}
    var_2 = module_0.remove_entity_headers(dict_1)
    list_0 = [var_2]
    complex_0 = 1929.26 - 1582.8373j
    tuple_0 = (list_0, complex_0)
    var_3 = module_1.ismodule(tuple_0)
    str_1 = var_3.__str__()
    int_0 = -1651
    var_4 = module_0.has_message_body(int_0)


def test_case_11():
    str_0 = "content-md5"
    list_0 = [str_0, str_0]
    dict_0 = {str_0: list_0, str_0: str_0, str_0: list_0}
    var_0 = module_0.remove_entity_headers(dict_0)
    dict_1 = {str_0: dict_0}
    var_1 = module_0.remove_entity_headers(dict_1, dict_1)
    module_0.is_entity_header(var_1)
