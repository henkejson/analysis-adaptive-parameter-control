# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.helpers as module_0


def test_case_0():
    bool_0 = module_0.is_atty()
    var_0 = module_0.has_message_body(bool_0)


def test_case_1():
    bool_0 = module_0.is_atty()
    str_0 = '=N"4Y:?NgZHb2'
    module_0.remove_entity_headers(str_0)


def test_case_2():
    bool_0 = module_0.is_atty()


def test_case_3():
    default_0 = module_0.Default()


def test_case_4():
    bool_0 = False
    str_0 = bool_0.__str__()
    dict_0 = {str_0: str_0}
    var_0 = module_0.remove_entity_headers(dict_0)


def test_case_5():
    default_0 = module_0.Default()
    str_0 = "|pEv"
    var_0 = module_0.is_hop_by_hop_header(str_0)
    bool_0 = module_0.is_atty()


def test_case_6():
    list_0 = []
    dict_0 = {}
    default_0 = module_0.Default(*list_0, **dict_0)
    var_0 = default_0.__repr__()


def test_case_7():
    bool_0 = module_0.is_atty()
    default_0 = module_0.Default()
    str_0 = "sanic.server"
    var_0 = module_0.import_string(str_0)
    bool_1 = True
    var_1 = module_0.has_message_body(bool_1)
    str_1 = default_0.__str__()
    str_2 = var_1.__str__()
    str_3 = var_1.__str__()
    bool_2 = module_0.is_atty()
    dict_0 = {str_3: var_1}
    var_2 = module_0.remove_entity_headers(dict_0)
    module_0.is_hop_by_hop_header(var_1)


def test_case_8():
    dict_0 = {}
    default_0 = module_0.Default()
    var_0 = module_0.remove_entity_headers(dict_0)
    module_0.import_string(dict_0)


def test_case_9():
    float_0 = 1845.249
    var_0 = module_0.has_message_body(float_0)
    module_0.import_string(var_0)


def test_case_10():
    int_0 = 304
    var_0 = module_0.has_message_body(int_0)
    bool_0 = True
    str_0 = var_0.__str__()
    str_1 = var_0.__str__()
    str_2 = bool_0.__str__()
    bool_1 = module_0.is_atty()
    dict_0 = {str_2: str_0}
    var_1 = module_0.remove_entity_headers(dict_0)
    var_2 = module_0.has_message_body(bool_1)
    var_3 = module_0.has_message_body(int_0)
    module_0.is_entity_header(var_1)


def test_case_11():
    str_0 = "sanic.server"
    var_0 = module_0.import_string(str_0)
    default_0 = module_0.Default()
    bool_0 = module_0.is_atty()
    var_1 = module_0.has_message_body(bool_0)
