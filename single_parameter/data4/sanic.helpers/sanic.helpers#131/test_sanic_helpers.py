# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.helpers as module_0


def test_case_0():
    int_0 = 20
    var_0 = module_0.has_message_body(int_0)


def test_case_1():
    none_type_0 = None
    module_0.remove_entity_headers(none_type_0)


def test_case_2():
    bool_0 = module_0.is_atty()


def test_case_3():
    none_type_0 = None
    module_0.is_entity_header(none_type_0)


def test_case_4():
    complex_0 = -2066.891 - 576.2833j
    module_0.is_hop_by_hop_header(complex_0)


def test_case_5():
    bool_0 = False
    dict_0 = {bool_0: bool_0, bool_0: bool_0}
    default_0 = module_0.Default()
    str_0 = default_0.__str__()
    bool_1 = module_0.is_atty()
    var_0 = default_0.__repr__()
    module_0.remove_entity_headers(dict_0, var_0)


def test_case_6():
    int_0 = 482
    var_0 = module_0.has_message_body(int_0)
    bool_0 = module_0.is_atty()
    dict_0 = {var_0: var_0, var_0: var_0}
    module_0.remove_entity_headers(dict_0)


def test_case_7():
    bool_0 = False
    dict_0 = {bool_0: bool_0, bool_0: bool_0}
    module_0.remove_entity_headers(dict_0)


def test_case_8():
    bytes_0 = b""
    dict_0 = {bytes_0: bytes_0}
    var_0 = module_0.remove_entity_headers(dict_0)


def test_case_9():
    int_0 = 204
    var_0 = module_0.has_message_body(int_0)
    str_0 = var_0.__str__()
    int_1 = 322
    var_1 = module_0.has_message_body(int_1)
    bool_0 = True
    str_1 = module_0.has_message_body(bool_0)
    var_2 = var_1.__repr__()
    bool_1 = module_0.is_atty()
    var_3 = var_1.__repr__()
    dict_0 = {var_2: int_1, bool_0: bool_0, bool_0: bool_0}
    var_4 = bool_1.__repr__()
    default_0 = module_0.Default()
    var_5 = default_0.__repr__()
    var_6 = module_0.has_message_body(bool_0)
    var_7 = module_0.is_hop_by_hop_header(var_4)
    str_2 = var_5.__str__()
    str_3 = dict_0.__str__()
    module_0.remove_entity_headers(dict_0)
