# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import py_backwards.utils.helpers as module_0
import re as module_1


def test_case_0():
    none_type_0 = None
    none_type_1 = module_0.debug(none_type_0)


def test_case_1():
    variables_generator_0 = module_0.VariablesGenerator()


def test_case_2():
    variables_generator_0 = module_0.VariablesGenerator()
    none_type_0 = None
    none_type_1 = module_0.debug(none_type_0)
    callable_0 = module_0.eager(none_type_0)


def test_case_3():
    none_type_0 = None
    none_type_1 = module_0.debug(none_type_0)
    callable_0 = module_0.eager(none_type_0)
    module_0.get_source(callable_0)


def test_case_4():
    str_0 = "Q.Zh"
    none_type_0 = module_0.debug(str_0)
    none_type_1 = module_0.warn(str_0)
    none_type_2 = module_0.warn(str_0)
    callable_0 = module_0.eager(none_type_1)
    none_type_3 = module_0.warn(none_type_1)
    str_1 = "O~Fvg.*,2'\x0b$"
    none_type_4 = module_0.warn(str_1)
    module_1.fullmatch(none_type_1, none_type_0, str_0)


def test_case_5():
    none_type_0 = None
    none_type_1 = module_0.debug(none_type_0)
    callable_0 = module_0.eager(none_type_0)
    callable_0.__call__(none_type_1, none_type_1, qualname=callable_0)
