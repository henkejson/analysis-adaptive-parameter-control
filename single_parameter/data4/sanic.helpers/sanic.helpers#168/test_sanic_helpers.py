# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.helpers as module_0


def test_case_0():
    bool_0 = module_0.is_atty()
    int_0 = 340
    var_0 = module_0.has_message_body(int_0)


def test_case_1():
    int_0 = 1048
    str_0 = int_0.__str__()
    dict_0 = {str_0: int_0}
    var_0 = module_0.remove_entity_headers(dict_0)


def test_case_2():
    bool_0 = module_0.is_atty()


def test_case_3():
    bool_0 = module_0.is_atty()
    module_0.is_entity_header(bool_0)


def test_case_4():
    bool_0 = module_0.is_atty()
    module_0.is_hop_by_hop_header(bool_0)


def test_case_5():
    bytes_0 = b"\xfc\xa9\xf2\x87\x08\xa1\xd8\xff\x93\xa1^\x1co\xb1\xcd"
    dict_0 = {}
    default_0 = module_0.Default(**dict_0)
    str_0 = default_0.__str__()
    var_0 = module_0.is_hop_by_hop_header(bytes_0)
    bool_0 = module_0.is_atty()


def test_case_6():
    int_0 = -1657
    var_0 = module_0.has_message_body(int_0)
    var_1 = var_0.__repr__()
    module_0.remove_entity_headers(var_1)


def test_case_7():
    bool_0 = module_0.is_atty()
    int_0 = 304
    var_0 = module_0.has_message_body(int_0)
