# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.helpers as module_0


def test_case_0():
    dict_0 = {}
    list_0 = []
    default_0 = module_0.Default(*list_0)
    module_0.has_message_body(dict_0)


def test_case_1():
    dict_0 = {}
    var_0 = module_0.remove_entity_headers(dict_0)


def test_case_2():
    bool_0 = module_0.is_atty()


def test_case_3():
    dict_0 = {}
    module_0.is_entity_header(dict_0)


def test_case_4():
    default_0 = module_0.Default()
    module_0.is_hop_by_hop_header(default_0)


def test_case_5():
    default_0 = module_0.Default()
    var_0 = default_0.__repr__()
    list_0 = [default_0]
    module_0.Default(*list_0)


def test_case_6():
    dict_0 = {}
    var_0 = module_0.remove_entity_headers(dict_0, dict_0)
    var_1 = module_0.Default()
    str_0 = var_1.__str__()
    module_0.has_message_body(var_0)


def test_case_7():
    bool_0 = module_0.is_atty()
    var_0 = module_0.has_message_body(bool_0)
    bool_1 = module_0.is_atty()
    var_1 = var_0.__repr__()


def test_case_8():
    int_0 = 332
    var_0 = module_0.has_message_body(int_0)
    var_1 = var_0.__repr__()
    var_2 = var_1.__repr__()


def test_case_9():
    dict_0 = {}
    var_0 = module_0.remove_entity_headers(dict_0)
    bool_0 = module_0.is_atty()
    default_0 = module_0.Default()
    str_0 = default_0.__str__()
    int_0 = 304
    var_1 = module_0.has_message_body(int_0)
    str_1 = var_0.__str__()
    str_2 = default_0.__str__()
    str_3 = default_0.__str__()
    var_2 = module_0.is_entity_header(str_1)


def test_case_10():
    bool_0 = module_0.is_atty()
    dict_0 = {bool_0: bool_0, bool_0: bool_0, bool_0: bool_0, bool_0: bool_0}
    module_0.remove_entity_headers(dict_0)


def test_case_11():
    bytes_0 = b"\xf6\xeb\xc9"
    dict_0 = {bytes_0: bytes_0, bytes_0: bytes_0, bytes_0: bytes_0, bytes_0: bytes_0}
    var_0 = module_0.remove_entity_headers(dict_0)
