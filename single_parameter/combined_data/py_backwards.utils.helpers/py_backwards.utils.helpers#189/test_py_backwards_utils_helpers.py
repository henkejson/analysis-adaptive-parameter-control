# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import py_backwards.utils.helpers as module_0
import enum as module_1
import re as module_2


def test_case_0():
    none_type_0 = None
    none_type_1 = module_0.debug(none_type_0)


def test_case_1():
    variables_generator_0 = module_0.VariablesGenerator()


def test_case_2():
    none_type_0 = None
    callable_0 = module_0.eager(none_type_0)


def test_case_3():
    bytes_0 = b"\x14\n"
    none_type_0 = module_0.warn(bytes_0)
    auto_0 = module_1.auto()
    none_type_1 = module_0.warn(auto_0)
    module_0.get_source(none_type_1)


def test_case_4():
    str_0 = "\\e@|\r:t!3+"
    none_type_0 = module_0.warn(str_0)


def test_case_5():
    bytes_0 = b"\xf7\xa2\x97\xe3\xec\xcd\xe3\xd3\xb5\xdd\xc7\\\xce\x9d\xf8"
    variables_generator_0 = module_0.VariablesGenerator()
    none_type_0 = module_0.warn(bytes_0)
    var_0 = module_2.purge()
    none_type_1 = module_0.debug(var_0)
    none_type_2 = module_0.warn(none_type_1)
    callable_0 = module_0.eager(var_0)
    var_1 = var_0.__dir__()
    none_type_3 = module_0.debug(bytes_0)
    none_type_4 = module_0.debug(bytes_0)
    callable_1 = module_0.eager(var_1)
    tuple_0 = (bytes_0,)
    callable_2 = module_0.eager(var_0)
    var_2 = var_0.__bool__()
    callable_2.__call__(bytes_0, none_type_1, qualname=tuple_0)
