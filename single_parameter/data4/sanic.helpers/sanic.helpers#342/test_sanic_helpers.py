# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.helpers as module_0


def test_case_0():
    none_type_0 = None
    module_0.has_message_body(none_type_0)


def test_case_1():
    bytes_0 = b"\x84\x95rgrT\xc4{\xd6&~\x05\xdf"
    module_0.remove_entity_headers(bytes_0)


def test_case_2():
    bool_0 = module_0.is_atty()


def test_case_3():
    default_0 = module_0.Default()


def test_case_4():
    float_0 = -2423.140904
    dict_0 = {float_0: float_0, float_0: float_0, float_0: float_0, float_0: float_0}
    module_0.remove_entity_headers(dict_0)


def test_case_5():
    bool_0 = False
    module_0.is_hop_by_hop_header(bool_0)


def test_case_6():
    default_0 = module_0.Default()
    var_0 = default_0.__repr__()
    dict_0 = {var_0: var_0, var_0: var_0, var_0: var_0}
    var_1 = module_0.remove_entity_headers(dict_0)


def test_case_7():
    default_0 = module_0.Default()
    var_0 = default_0.__repr__()
    dict_0 = {var_0: var_0, var_0: var_0, var_0: var_0}
    str_0 = default_0.__str__()
    var_1 = default_0.__repr__()
    var_2 = module_0.remove_entity_headers(dict_0)


def test_case_8():
    int_0 = 1
    var_0 = module_0.has_message_body(int_0)
    str_0 = var_0.__str__()


def test_case_9():
    int_0 = 100
    var_0 = module_0.has_message_body(int_0)
    bool_0 = module_0.is_atty()
