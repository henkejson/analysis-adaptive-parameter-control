# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.helpers as module_0
import builtins as module_1


def test_case_0():
    none_type_0 = None
    module_0.has_message_body(none_type_0)


def test_case_1():
    str_0 = "<input>["
    dict_0 = {str_0: str_0}
    var_0 = module_0.remove_entity_headers(dict_0)
    module_0.has_message_body(dict_0)


def test_case_2():
    bool_0 = module_0.is_atty()


def test_case_3():
    default_0 = module_0.Default()


def test_case_4():
    none_type_0 = None
    module_0.is_hop_by_hop_header(none_type_0)


def test_case_5():
    dict_0 = {}
    var_0 = module_0.remove_entity_headers(dict_0)
    default_0 = module_0.Default(*var_0)
    str_0 = default_0.__str__()
    bool_0 = module_0.is_atty()
    int_0 = 491
    var_1 = module_0.has_message_body(int_0)


def test_case_6():
    int_0 = 390
    var_0 = module_0.has_message_body(int_0)


def test_case_7():
    bool_0 = True
    var_0 = module_0.has_message_body(bool_0)


def test_case_8():
    str_0 = "iput"
    var_0 = module_0.is_entity_header(str_0)
    dict_0 = {str_0: str_0}
    object_0 = module_1.object()
    var_1 = module_0.remove_entity_headers(dict_0)
    var_2 = var_0.__repr__()
    default_0 = module_0.Default()
    var_3 = module_0.has_message_body(var_0)
    bool_0 = module_0.is_atty()
    str_1 = var_0.__str__()
    str_2 = default_0.__str__()
    str_3 = var_3.__str__()
    int_0 = 304
    str_4 = str_1.__str__()
    var_4 = var_2.__repr__()
    var_5 = module_0.has_message_body(int_0)
    module_0.is_hop_by_hop_header(var_3)


def test_case_9():
    str_0 = "Content-Type"
    dict_0 = {str_0: str_0}
    var_0 = dict_0.__repr__()
    str_1 = str_0.__str__()
    str_2 = var_0.__str__()
    bool_0 = True
    str_3 = module_0.remove_entity_headers(dict_0)
    module_0.is_hop_by_hop_header(bool_0)


def test_case_10():
    str_0 = "Content-Type"
    var_0 = module_0.is_entity_header(str_0)
    dict_0 = {str_0: str_0}
    object_0 = module_0.remove_entity_headers(dict_0, dict_0)
    var_1 = module_0.remove_entity_headers(dict_0)
    var_2 = var_0.__repr__()
    default_0 = module_0.Default()
    var_3 = module_0.has_message_body(var_0)
    bool_0 = module_0.is_atty()
    str_1 = default_0.__str__()
    module_0.import_string(var_0, str_1)
