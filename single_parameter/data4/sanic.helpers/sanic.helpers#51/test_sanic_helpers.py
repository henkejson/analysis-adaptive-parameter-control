# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.helpers as module_0


def test_case_0():
    bool_0 = True
    var_0 = module_0.has_message_body(bool_0)


def test_case_1():
    float_0 = -4208.736642
    module_0.remove_entity_headers(float_0)


def test_case_2():
    bool_0 = module_0.is_atty()


def test_case_3():
    bool_0 = False
    module_0.is_entity_header(bool_0)


def test_case_4():
    none_type_0 = None
    module_0.is_hop_by_hop_header(none_type_0)


def test_case_5():
    list_0 = []
    default_0 = module_0.Default(*list_0)
    var_0 = default_0.__repr__()
    module_0.has_message_body(var_0)


def test_case_6():
    default_0 = module_0.Default()
    str_0 = default_0.__str__()
    module_0.has_message_body(default_0)


def test_case_7():
    int_0 = 997
    var_0 = module_0.has_message_body(int_0)
    module_0.remove_entity_headers(var_0)


def test_case_8():
    complex_0 = 863.952 + 2577.87j
    dict_0 = {complex_0: complex_0}
    module_0.remove_entity_headers(dict_0)


def test_case_9():
    str_0 = "]S$IZ0r]@;MfR1J:N'"
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0}
    var_0 = module_0.remove_entity_headers(dict_0)


def test_case_10():
    float_0 = 304.0
    var_0 = module_0.has_message_body(float_0)
    default_0 = module_0.Default()
    bool_0 = module_0.is_atty()
    str_0 = default_0.__str__()
    bool_1 = module_0.is_atty()
    var_1 = var_0.__repr__()
    str_1 = default_0.__str__()
    bool_2 = module_0.is_atty()
    none_type_0 = None
    module_0.has_message_body(none_type_0)
