# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import re as module_0
import py_backwards.utils.helpers as module_1


def test_case_0():
    var_0 = module_0.purge()
    none_type_0 = module_1.debug(var_0)


def test_case_1():
    variables_generator_0 = module_1.VariablesGenerator()


def test_case_2():
    str_0 = "\x0c\n-}HzxNoyeT"
    callable_0 = module_1.eager(str_0)
    none_type_0 = module_1.warn(str_0)
    callable_0.__call__(callable_0, none_type_0)


def test_case_3():
    none_type_0 = None
    module_1.get_source(none_type_0)


def test_case_4():
    var_0 = module_0.purge()
    none_type_0 = module_1.warn(var_0)
