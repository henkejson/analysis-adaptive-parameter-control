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
    variables_generator_0 = module_1.purge()
    callable_0 = module_0.eager(variables_generator_0)


def test_case_3():
    dict_0 = {}
    module_0.get_source(dict_0)


def test_case_4():
    str_0 = "Tix"
    none_type_0 = module_0.warn(str_0)
    var_0 = module_1.compile(str_0)
    callable_0 = module_0.eager(var_0)


def test_case_5():
    str_0 = "7a{TKL"
    callable_0 = module_0.eager(str_0)
    module_1.sub(str_0, callable_0, str_0)
