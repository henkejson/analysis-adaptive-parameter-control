# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.helpers as module_0
import inspect as module_1
import builtins as module_2


def test_case_0():
    bytes_0 = b"U\x99"
    module_0.has_message_body(bytes_0)


def test_case_1():
    bool_0 = True
    module_0.remove_entity_headers(bool_0)


def test_case_2():
    bool_0 = module_0.is_atty()


def test_case_3():
    default_0 = module_0.Default()


def test_case_4():
    bool_0 = True
    module_0.is_entity_header(bool_0)


def test_case_5():
    dict_0 = {}
    var_0 = module_0.remove_entity_headers(dict_0)
    module_0.is_hop_by_hop_header(var_0)


def test_case_6():
    default_0 = module_0.Default()
    var_0 = default_0.__repr__()


def test_case_7():
    default_0 = module_0.Default()
    str_0 = default_0.__str__()


def test_case_8():
    int_0 = 2421
    var_0 = module_0.has_message_body(int_0)
    bool_0 = True
    module_0.is_entity_header(bool_0)


def test_case_9():
    default_0 = module_0.Default()
    default_1 = module_0.Default()
    var_0 = module_1.ismodule(default_1)
    str_0 = default_1.__str__()
    var_1 = module_0.has_message_body(var_0)
    var_2 = var_1.__repr__()
    default_2 = module_0.Default()
    str_1 = var_1.__str__()
    module_0.has_message_body(default_1)


def test_case_10():
    dict_0 = {}
    var_0 = module_0.remove_entity_headers(dict_0)
    var_1 = module_1.ismodule(dict_0)


def test_case_11():
    str_0 = "o9<noC\x0c|o>9:cL%\r`U"
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0}
    var_0 = module_0.remove_entity_headers(dict_0)


def test_case_12():
    int_0 = 204
    bool_0 = module_0.is_atty()
    var_0 = module_0.has_message_body(int_0)
    default_0 = module_0.Default()
    object_0 = module_2.object()
    bool_1 = module_0.is_atty()
    str_0 = default_0.__str__()
    var_1 = module_0.is_entity_header(str_0)
    var_2 = module_0.has_message_body(var_1)
    var_3 = module_1.ismodule(var_1)
    module_0.remove_entity_headers(var_3)
