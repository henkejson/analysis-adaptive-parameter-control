# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.helpers as module_0


def test_case_0():
    bytes_0 = b'\xe6"\xb0\xb0\x1e\xac$\xfa_\xa8'
    module_0.has_message_body(bytes_0)


def test_case_1():
    bytes_0 = b"\xf7$\x9e\x88"
    dict_0 = {bytes_0: bytes_0, bytes_0: bytes_0}
    var_0 = module_0.remove_entity_headers(dict_0)


def test_case_2():
    dict_0 = {}
    var_0 = module_0.remove_entity_headers(dict_0, dict_0)


def test_case_3():
    bool_0 = module_0.is_atty()


def test_case_4():
    none_type_0 = None
    module_0.is_hop_by_hop_header(none_type_0)


def test_case_5():
    default_0 = module_0.Default()
    var_0 = default_0.__repr__()
    str_0 = default_0.__str__()
    var_1 = default_0.__repr__()
    bool_0 = module_0.is_atty()


def test_case_6():
    float_0 = 1.1
    var_0 = module_0.has_message_body(float_0)
    bool_0 = module_0.is_atty()


def test_case_7():
    float_0 = 603.7
    var_0 = module_0.has_message_body(float_0)
    bool_0 = module_0.is_atty()
    var_1 = var_0.__repr__()
    var_2 = var_1.__repr__()
    var_3 = var_1.__repr__()
    module_0.import_string(var_1)


def test_case_8():
    str_0 = "(\t6\t#Dzq2uM2W_kA`?y"
    var_0 = module_0.is_hop_by_hop_header(str_0)
    var_1 = var_0.__repr__()
    default_0 = module_0.Default()
    dict_0 = {var_1: var_0, str_0: default_0}
    var_2 = module_0.remove_entity_headers(dict_0)
    str_1 = var_0.__str__()
    var_3 = module_0.is_entity_header(var_1)
    var_4 = default_0.__repr__()
    var_5 = module_0.is_entity_header(var_1)
    bool_0 = module_0.is_atty()
    int_0 = 304
    var_6 = module_0.has_message_body(int_0)
    module_0.Default(*var_3)
