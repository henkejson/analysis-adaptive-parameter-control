# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.helpers as module_0
import inspect as module_1


def test_case_0():
    float_0 = 2462.927
    var_0 = module_0.has_message_body(float_0)


def test_case_1():
    bool_0 = True
    module_0.remove_entity_headers(bool_0)


def test_case_2():
    str_0 = "p fe!Df]mw;?q\\tE${V"
    dict_0 = {str_0: str_0}
    var_0 = module_0.remove_entity_headers(dict_0)


def test_case_3():
    bool_0 = module_0.is_atty()


def test_case_4():
    bool_0 = module_0.is_atty()
    module_0.is_entity_header(bool_0)


def test_case_5():
    float_0 = 2462.927
    module_0.is_hop_by_hop_header(float_0)


def test_case_6():
    default_0 = module_0.Default()
    var_0 = default_0.__repr__()
    dict_0 = {var_0: var_0, var_0: var_0, var_0: var_0}
    var_1 = module_0.remove_entity_headers(dict_0)


def test_case_7():
    float_0 = 203.75033216697386
    var_0 = module_0.has_message_body(float_0)
    var_1 = module_1.ismodule(var_0)
    bytes_0 = b"R"
    dict_0 = {bytes_0: bytes_0, bytes_0: bytes_0, bytes_0: bytes_0}
    var_2 = module_0.remove_entity_headers(dict_0)
    var_3 = var_1.__repr__()
    default_0 = module_0.Default()
    var_4 = default_0.__repr__()
    var_5 = var_2.__repr__()
    bool_0 = module_0.is_atty()
    var_6 = var_1.__repr__()
    str_0 = var_1.__str__()
    str_1 = default_0.__str__()
    module_0.is_hop_by_hop_header(dict_0)


def test_case_8():
    bool_0 = True
    var_0 = module_0.has_message_body(bool_0)


def test_case_9():
    float_0 = 204.0
    var_0 = module_0.has_message_body(float_0)
    var_1 = module_1.ismodule(var_0)
    bytes_0 = b"R"
    dict_0 = {bytes_0: bytes_0, bytes_0: bytes_0, bytes_0: bytes_0}
    var_2 = var_1.__repr__()
    var_3 = dict_0.__repr__()
    var_4 = var_3.__repr__()
    bool_0 = module_0.is_atty()
    var_5 = module_1.ismodule(float_0)
    module_0.has_message_body(bytes_0)
