# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import py_backwards.utils.helpers as module_0
import re as module_1


def test_case_0():
    str_0 = "urlunparse"
    none_type_0 = module_0.debug(str_0)


def test_case_1():
    variables_generator_0 = module_0.VariablesGenerator()


def test_case_2():
    str_0 = "7\\]"
    none_type_0 = module_0.debug(str_0)
    var_0 = module_1.purge()
    callable_0 = module_0.eager(none_type_0)
    none_type_1 = module_0.debug(callable_0)
    regex_flag_0 = module_1.RegexFlag.DEBUG
    none_type_2 = module_0.debug(regex_flag_0)
    variables_generator_0 = module_0.VariablesGenerator()
    callable_1 = module_0.eager(variables_generator_0)


def test_case_3():
    none_type_0 = None
    module_0.get_source(none_type_0)


def test_case_4():
    variables_generator_0 = module_0.VariablesGenerator()
    str_0 = "As&47oy\nq\x0c4"
    none_type_0 = module_0.warn(str_0)


def test_case_5():
    str_0 = "g<w"
    none_type_0 = None
    none_type_1 = module_0.debug(none_type_0)
    variables_generator_0 = module_0.VariablesGenerator()
    none_type_2 = module_0.warn(none_type_1)
    none_type_3 = module_0.warn(str_0)
    none_type_4 = module_0.warn(str_0)
    callable_0 = module_0.eager(str_0)
    callable_0.__call__(variables_generator_0, none_type_3, type=none_type_3)
