# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.helpers as module_0


def test_case_0():
    int_0 = 313
    var_0 = module_0.has_message_body(int_0)
    module_0.Default(*var_0, **var_0)


def test_case_1():
    str_0 = ""
    list_0 = [str_0]
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0, str_0: list_0}
    var_0 = module_0.remove_entity_headers(dict_0, list_0)


def test_case_2():
    str_0 = ""
    list_0 = []
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0, str_0: list_0}
    var_0 = module_0.remove_entity_headers(dict_0, list_0)


def test_case_3():
    bool_0 = module_0.is_atty()


def test_case_4():
    none_type_0 = None
    module_0.is_entity_header(none_type_0)


def test_case_5():
    float_0 = -441.111269
    module_0.is_hop_by_hop_header(float_0)


def test_case_6():
    bool_0 = module_0.is_atty()
    default_0 = module_0.Default()
    var_0 = default_0.__repr__()
    var_1 = module_0.has_message_body(bool_0)
    var_2 = var_1.__repr__()
    str_0 = var_1.__str__()
    var_3 = module_0.has_message_body(var_1)
    int_0 = 1657
    dict_0 = {str_0: int_0}
    str_1 = var_2.__str__()
    var_4 = module_0.remove_entity_headers(dict_0)
    module_0.is_hop_by_hop_header(var_4)


def test_case_7():
    bool_0 = module_0.is_atty()
    var_0 = module_0.has_message_body(bool_0)
    var_1 = module_0.has_message_body(var_0)
    default_0 = module_0.Default()
    dict_0 = {}
    var_2 = module_0.remove_entity_headers(dict_0)
    str_0 = default_0.__str__()
    bool_1 = module_0.is_atty()
    var_3 = default_0.__repr__()
    module_0.import_string(str_0)


def test_case_8():
    int_0 = 308
    var_0 = module_0.has_message_body(int_0)
    default_0 = module_0.has_message_body(var_0)
    str_0 = var_0.__str__()
    var_1 = var_0.__repr__()
    str_1 = var_1.__str__()


def test_case_9():
    int_0 = 304
    var_0 = module_0.has_message_body(int_0)
    module_0.is_entity_header(var_0)
