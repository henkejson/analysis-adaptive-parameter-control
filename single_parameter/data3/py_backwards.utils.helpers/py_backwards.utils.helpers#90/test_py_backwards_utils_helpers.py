# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import py_backwards.utils.helpers as module_0
import py_backwards.exceptions as module_1
import re as module_2


def test_case_0():
    variables_generator_0 = module_0.VariablesGenerator()
    none_type_0 = module_0.debug(variables_generator_0)


def test_case_1():
    variables_generator_0 = module_0.VariablesGenerator()


def test_case_2():
    none_type_0 = None
    callable_0 = module_0.eager(none_type_0)
    none_type_1 = module_0.debug(callable_0)


def test_case_3():
    str_0 = "m.4TL4{V'J"
    module_0.get_source(str_0)


def test_case_4():
    variables_generator_0 = module_0.VariablesGenerator()
    none_type_0 = module_0.warn(variables_generator_0)


def test_case_5():
    none_type_0 = None
    callable_0 = module_0.eager(none_type_0)
    str_0 = "B<"
    none_type_1 = module_0.warn(str_0)
    str_1 = "$|\x0coDIqjyrIO=f8"
    int_0 = -1842
    none_type_2 = module_0.warn(str_0)
    compilation_error_0 = module_1.CompilationError(str_0, str_1, int_0, str_1)
    callable_1 = module_0.eager(str_1)
    callable_2 = module_0.eager(none_type_0)
    module_2.sub(str_1, callable_1, str_1)
