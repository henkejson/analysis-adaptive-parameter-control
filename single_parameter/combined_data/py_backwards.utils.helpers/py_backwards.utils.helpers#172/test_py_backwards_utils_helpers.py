# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import py_backwards.utils.helpers as module_0
import re as module_1


def test_case_0():
    bool_0 = False
    none_type_0 = module_0.debug(bool_0)


def test_case_1():
    variables_generator_0 = module_0.VariablesGenerator()


def test_case_2():
    none_type_0 = None
    callable_0 = module_0.eager(none_type_0)
    none_type_1 = module_0.debug(callable_0)
    none_type_2 = module_0.debug(callable_0)
    variables_generator_0 = module_0.VariablesGenerator()
    none_type_3 = module_0.debug(callable_0)


def test_case_3():
    none_type_0 = None
    none_type_1 = module_0.debug(none_type_0)
    var_0 = module_1.RegexFlag.IGNORECASE
    module_0.get_source(var_0)


def test_case_4():
    none_type_0 = None
    none_type_1 = module_0.debug(none_type_0)
    none_type_2 = module_0.warn(none_type_0)


def test_case_5():
    none_type_0 = None
    callable_0 = module_0.eager(none_type_0)
    none_type_1 = module_0.debug(callable_0)
    callable_1 = module_0.eager(callable_0)
    callable_1.__call__(none_type_0, none_type_1, callable_1)
