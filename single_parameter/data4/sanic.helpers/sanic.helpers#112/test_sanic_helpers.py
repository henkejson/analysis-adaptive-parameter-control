# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.helpers as module_0
import inspect as module_1


def test_case_0():
    int_0 = 16
    var_0 = module_0.has_message_body(int_0)


def test_case_1():
    bytes_0 = b"\xb7\xbd\xa8\x84S\x93\xc2\xe05\xbf"
    dict_0 = {bytes_0: bytes_0, bytes_0: bytes_0, bytes_0: bytes_0}
    var_0 = module_0.remove_entity_headers(dict_0)


def test_case_2():
    bool_0 = module_0.is_atty()


def test_case_3():
    str_0 = "[o`eK'$<:NFZlpjvcrnA"
    set_0 = {str_0, str_0}
    module_0.is_entity_header(set_0)


def test_case_4():
    bool_0 = module_0.is_atty()
    var_0 = module_0.has_message_body(bool_0)
    module_0.is_hop_by_hop_header(bool_0)


def test_case_5():
    bool_0 = False
    default_0 = module_0.Default()
    dict_0 = {bool_0: bool_0, bool_0: bool_0, bool_0: bool_0, bool_0: bool_0}
    var_0 = default_0.__repr__()
    module_0.import_string(dict_0, bool_0)


def test_case_6():
    float_0 = 781.1716
    var_0 = module_0.has_message_body(float_0)
    var_1 = var_0.__repr__()
    default_0 = module_0.Default()
    str_0 = default_0.__str__()
    bool_0 = module_0.is_atty()
    var_2 = default_0.__repr__()
    var_3 = var_2.__repr__()
    var_4 = var_3.__repr__()
    module_0.is_hop_by_hop_header(bool_0)


def test_case_7():
    int_0 = 425
    var_0 = module_0.has_message_body(int_0)


def test_case_8():
    int_0 = 304
    var_0 = module_0.has_message_body(int_0)
    bool_0 = module_0.is_atty()
    default_0 = module_0.Default()
    dict_0 = {default_0: default_0, default_0: default_0}
    str_0 = bool_0.__str__()
    str_1 = module_1.ismodule(var_0)
    var_1 = dict_0.__repr__()
    module_0.remove_entity_headers(dict_0)
