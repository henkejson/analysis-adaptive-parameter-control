# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.helpers as module_0


def test_case_0():
    int_0 = 39
    var_0 = module_0.has_message_body(int_0)


def test_case_1():
    dict_0 = {}
    var_0 = module_0.remove_entity_headers(dict_0)


def test_case_2():
    bool_0 = module_0.is_atty()


def test_case_3():
    default_0 = module_0.Default()
    module_0.is_entity_header(default_0)


def test_case_4():
    bool_0 = module_0.is_atty()
    module_0.is_hop_by_hop_header(bool_0)


def test_case_5():
    default_0 = module_0.Default()
    str_0 = default_0.__str__()
    int_0 = 302
    var_0 = module_0.has_message_body(int_0)


def test_case_6():
    int_0 = 302
    var_0 = module_0.has_message_body(int_0)


def test_case_7():
    dict_0 = {}
    str_0 = dict_0.__str__()
    dict_1 = {str_0: dict_0}
    var_0 = module_0.remove_entity_headers(dict_1)


def test_case_8():
    int_0 = 114
    dict_0 = {}
    default_0 = module_0.Default(**dict_0)
    var_0 = module_0.Default()
    str_0 = int_0.__str__()
    str_1 = default_0.__str__()
    str_2 = dict_0.__str__()
    dict_1 = {str_1: str_2}
    bool_0 = module_0.is_atty()
    bool_1 = module_0.is_atty()
    var_1 = module_0.is_hop_by_hop_header(str_1)
    str_3 = dict_1.__str__()
    str_4 = default_0.__str__()
    var_2 = module_0.remove_entity_headers(dict_1)
    var_3 = default_0.__repr__()
    var_4 = module_0.has_message_body(bool_1)
    int_1 = 204
    var_5 = module_0.has_message_body(int_1)
    module_0.import_string(var_1, bool_1)
