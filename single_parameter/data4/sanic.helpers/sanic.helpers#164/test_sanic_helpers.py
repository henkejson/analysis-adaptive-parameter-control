# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.helpers as module_0


def test_case_0():
    none_type_0 = None
    module_0.has_message_body(none_type_0)


def test_case_1():
    dict_0 = {}
    var_0 = module_0.remove_entity_headers(dict_0)


def test_case_2():
    bool_0 = module_0.is_atty()


def test_case_3():
    none_type_0 = None
    module_0.is_entity_header(none_type_0)


def test_case_4():
    bool_0 = module_0.is_atty()
    module_0.is_hop_by_hop_header(bool_0)


def test_case_5():
    default_0 = module_0.Default()
    str_0 = default_0.__str__()
    str_1 = default_0.__str__()
    var_0 = default_0.__repr__()


def test_case_6():
    float_0 = 320.64
    var_0 = module_0.has_message_body(float_0)


def test_case_7():
    bool_0 = True
    var_0 = module_0.has_message_body(bool_0)
    var_1 = var_0.__repr__()
    dict_0 = {var_1: var_1, var_1: var_1, var_1: var_1, var_1: var_1}
    var_2 = module_0.remove_entity_headers(dict_0)


def test_case_8():
    int_0 = 633
    dict_0 = {int_0: int_0, int_0: int_0, int_0: int_0, int_0: int_0}
    module_0.remove_entity_headers(dict_0)


def test_case_9():
    bytes_0 = b"6\xdc(^\x9b\xb6\xbc"
    dict_0 = {bytes_0: bytes_0}
    var_0 = module_0.remove_entity_headers(dict_0)


def test_case_10():
    str_0 = "allow"
    dict_0 = {str_0: str_0}
    var_0 = module_0.remove_entity_headers(dict_0)
    bool_0 = module_0.is_atty()
    default_0 = module_0.Default()
    default_1 = module_0.Default()
    complex_0 = 1007 + 1250.1608j
    set_0 = {bool_0, bool_0, complex_0}
    str_1 = var_0.__str__()
    str_2 = default_0.__str__()
    module_0.is_hop_by_hop_header(set_0)


def test_case_11():
    bytes_0 = b"\xeaI\xa5\xcf\xebC\xe8khT\x8f\xbf"
    var_0 = module_0.is_hop_by_hop_header(bytes_0)
    var_1 = var_0.__repr__()
    str_0 = var_0.__str__()
    var_2 = var_0.__repr__()
    bool_0 = module_0.is_atty()
    int_0 = 304
    var_3 = module_0.has_message_body(int_0)
    var_4 = module_0.has_message_body(var_3)
    dict_0 = {
        var_1: var_0,
        bytes_0: bytes_0,
        bool_0: bytes_0,
        int_0: int_0,
        bool_0: var_1,
        var_1: var_1,
    }
    bool_1 = module_0.is_atty()
    module_0.remove_entity_headers(dict_0)


def test_case_12():
    bool_0 = module_0.is_atty()
    var_0 = module_0.has_message_body(bool_0)
    str_0 = "allow"
    dict_0 = {str_0: str_0}
    var_1 = module_0.remove_entity_headers(dict_0, dict_0)
    var_2 = var_0.__repr__()
    default_0 = module_0.Default()
    str_1 = default_0.__str__()
