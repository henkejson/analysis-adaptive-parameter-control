# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import py_backwards.utils.helpers as module_0
import locale as module_1
import re as module_2


def test_case_0():
    str_0 = "rP;oad"
    none_type_0 = module_0.debug(str_0)


def test_case_1():
    variables_generator_0 = module_0.VariablesGenerator()


def test_case_2():
    str_0 = ":u0"
    variables_generator_0 = module_0.VariablesGenerator()
    none_type_0 = module_0.warn(str_0)
    error_0 = module_1.Error()
    callable_0 = module_0.eager(error_0)


def test_case_3():
    str_0 = "g_]gG+S`{m\n2\x0b-q>"
    module_0.get_source(str_0)


def test_case_4():
    str_0 = ":u0"
    variables_generator_0 = module_0.VariablesGenerator()
    none_type_0 = module_0.warn(str_0)
    error_0 = module_1.Error()


def test_case_5():
    str_0 = "%"
    callable_0 = module_0.eager(str_0)
    none_type_0 = module_0.warn(str_0)
    module_2.sub(str_0, callable_0, str_0)
