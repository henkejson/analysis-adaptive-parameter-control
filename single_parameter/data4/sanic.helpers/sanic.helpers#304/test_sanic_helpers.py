# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.helpers as module_0
import inspect as module_1


def test_case_0():
    float_0 = 195.5792854095928
    var_0 = module_0.has_message_body(float_0)


def test_case_1():
    bytes_0 = b"4"
    module_0.remove_entity_headers(bytes_0)


def test_case_2():
    bool_0 = module_0.is_atty()


def test_case_3():
    str_0 = "n|10Wxr"
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0, str_0: str_0}
    var_0 = module_0.remove_entity_headers(dict_0)


def test_case_4():
    complex_0 = -456.8 - 1568.6911j
    module_0.is_hop_by_hop_header(complex_0)


def test_case_5():
    default_0 = module_0.Default()
    str_0 = "Iyrf\x0cG~;Rl*\x0b9t\x0cAd\r_]"
    var_0 = default_0.__repr__()
    var_1 = module_0.is_hop_by_hop_header(str_0)
    module_0.is_hop_by_hop_header(var_1)


def test_case_6():
    bool_0 = False
    int_0 = -1917
    var_0 = module_0.has_message_body(bool_0)
    tuple_0 = (int_0,)
    list_0 = [bool_0, tuple_0, int_0]
    var_1 = module_1.ismodule(list_0)
    default_0 = module_0.Default()
    str_0 = default_0.__str__()
    set_0 = {default_0, default_0, default_0}
    module_0.Default(**set_0)


def test_case_7():
    bool_0 = module_0.is_atty()
    float_0 = 175.066330840993
    var_0 = module_0.has_message_body(bool_0)
    var_1 = module_0.has_message_body(float_0)


def test_case_8():
    str_0 = "qeo4$"
    var_0 = module_0.is_hop_by_hop_header(str_0)
    bool_0 = module_0.is_hop_by_hop_header(str_0)
    float_0 = 204.0
    var_1 = module_0.has_message_body(float_0)
    var_2 = module_0.has_message_body(var_1)
    var_3 = module_0.has_message_body(float_0)
