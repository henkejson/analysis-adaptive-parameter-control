# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import py_backwards.utils.helpers as module_0
import re as module_1


def test_case_0():
    bool_0 = True
    none_type_0 = module_0.debug(bool_0)


def test_case_1():
    variables_generator_0 = module_0.VariablesGenerator()


def test_case_2():
    tuple_0 = ()
    callable_0 = module_0.eager(tuple_0)


def test_case_3():
    str_0 = "3l"
    none_type_0 = module_0.warn(str_0)
    var_0 = module_1.finditer(str_0, str_0)
    module_0.get_source(str_0)


def test_case_4():
    str_0 = "-d"
    str_1 = "a("
    none_type_0 = module_0.warn(str_1)
    none_type_1 = module_0.warn(str_0)


def test_case_5():
    bool_0 = True
    none_type_0 = module_0.warn(bool_0)
    callable_0 = module_0.eager(none_type_0)
    none_type_1 = module_0.debug(none_type_0)
    dict_0 = {}
    callable_1 = module_0.eager(none_type_0)
    variables_generator_0 = module_0.VariablesGenerator(**dict_0)
    none_type_2 = module_0.debug(bool_0)
    callable_0.__call__(none_type_1, none_type_0, qualname=bool_0, start=callable_0)
