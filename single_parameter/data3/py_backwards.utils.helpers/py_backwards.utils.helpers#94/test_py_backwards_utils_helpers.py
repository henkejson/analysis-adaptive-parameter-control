# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import py_backwards.utils.helpers as module_0
import re as module_1


def test_case_0():
    int_0 = -1270
    none_type_0 = module_0.debug(int_0)


def test_case_1():
    variables_generator_0 = module_0.VariablesGenerator()


def test_case_2():
    pattern_0 = module_1.purge()
    callable_0 = module_0.eager(pattern_0)


def test_case_3():
    int_0 = -1270
    module_0.get_source(int_0)


def test_case_4():
    str_0 = "E:`C{_\r4"
    none_type_0 = module_0.warn(str_0)
    none_type_1 = module_0.warn(str_0)
    module_1.escape(none_type_0)


def test_case_5():
    str_0 = "CacheFTPHandler"
    callable_0 = module_0.eager(str_0)
    variables_generator_0 = module_0.VariablesGenerator()
    none_type_0 = module_0.debug(str_0)
    none_type_1 = module_0.warn(str_0)
    var_0 = module_1.search(str_0, str_0)
    none_type_2 = module_0.debug(var_0)
    callable_1 = module_0.eager(var_0)
    none_type_3 = module_0.warn(var_0)
    none_type_4 = module_0.warn(none_type_0)
    callable_0.__call__(variables_generator_0, var_0, start=var_0)
