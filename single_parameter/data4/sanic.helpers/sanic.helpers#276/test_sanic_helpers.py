# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.helpers as module_0
import inspect as module_1


def test_case_0():
    float_0 = -925.28716
    var_0 = module_0.has_message_body(float_0)
    bool_0 = False
    module_0.import_string(bool_0, bool_0)


def test_case_1():
    str_0 = "k#8("
    dict_0 = {str_0: str_0}
    var_0 = module_0.remove_entity_headers(dict_0)


def test_case_2():
    bool_0 = module_0.is_atty()


def test_case_3():
    default_0 = module_0.Default()


def test_case_4():
    bool_0 = module_0.is_atty()
    str_0 = "/#p "
    var_0 = module_1.ismodule(str_0)
    str_1 = var_0.__str__()
    var_1 = module_0.is_hop_by_hop_header(str_0)
    var_2 = var_1.__repr__()
    var_3 = var_2.__repr__()


def test_case_5():
    bool_0 = module_0.is_atty()
    default_0 = module_0.Default()
    str_0 = default_0.__str__()


def test_case_6():
    str_0 = "zN,FL\x0bu*T2?fNs.4 5r"
    var_0 = module_0.is_entity_header(str_0)
    int_0 = 1076
    var_1 = module_0.has_message_body(int_0)
    var_2 = var_1.__repr__()
    dict_0 = {int_0: int_0, int_0: int_0, int_0: var_1, var_1: int_0}
    module_0.is_entity_header(dict_0)


def test_case_7():
    float_0 = 204.0
    var_0 = module_0.has_message_body(float_0)
    str_0 = var_0.__str__()
    str_1 = var_0.__str__()
    str_2 = var_0.__str__()
    bool_0 = module_0.is_atty()
    var_1 = module_0.has_message_body(var_0)
    default_0 = module_0.Default()
    var_2 = var_0.__repr__()
    var_3 = module_1.ismodule(str_0)
    var_4 = module_0.is_hop_by_hop_header(str_2)
    str_3 = default_0.__str__()
    module_0.remove_entity_headers(var_3)
