# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.helpers as module_0
import inspect as module_1


def test_case_0():
    bytes_0 = b""
    module_0.has_message_body(bytes_0)


def test_case_1():
    int_0 = 319
    var_0 = int_0.__repr__()
    dict_0 = {var_0: int_0}
    var_1 = module_0.remove_entity_headers(dict_0)


def test_case_2():
    bool_0 = module_0.is_atty()


def test_case_3():
    default_0 = module_0.Default()


def test_case_4():
    int_0 = 455
    var_0 = module_0.has_message_body(int_0)
    var_1 = module_0.has_message_body(var_0)
    module_0.is_entity_header(var_1)


def test_case_5():
    bytes_0 = b""
    var_0 = module_0.is_hop_by_hop_header(bytes_0)


def test_case_6():
    default_0 = module_0.Default()
    var_0 = default_0.__repr__()
    var_1 = module_0.is_entity_header(var_0)
    int_0 = 459
    var_2 = module_0.has_message_body(int_0)
    var_3 = module_0.has_message_body(var_2)
    module_0.is_hop_by_hop_header(var_3)


def test_case_7():
    bool_0 = True
    var_0 = module_0.has_message_body(bool_0)


def test_case_8():
    default_0 = module_0.Default()
    str_0 = default_0.__str__()
    var_0 = default_0.__repr__()
    var_1 = var_0.__repr__()
    str_1 = var_1.__str__()
    var_2 = default_0.__repr__()
    var_3 = default_0.__repr__()
    var_4 = var_3.__repr__()
    module_0.remove_entity_headers(var_3, str_0)


def test_case_9():
    int_0 = 455
    var_0 = module_0.has_message_body(int_0)
    var_1 = module_0.has_message_body(var_0)


def test_case_10():
    int_0 = 304
    var_0 = int_0.__repr__()
    dict_0 = {var_0: var_0}
    var_1 = module_0.remove_entity_headers(dict_0)
    bool_0 = module_0.is_atty()
    var_2 = module_1.ismodule(var_0)
    var_3 = module_1.ismodule(var_2)
    var_4 = var_3.__repr__()
    var_5 = module_0.has_message_body(int_0)
    var_6 = module_0.has_message_body(var_3)
    str_0 = var_3.__str__()
    module_0.import_string(var_3)
