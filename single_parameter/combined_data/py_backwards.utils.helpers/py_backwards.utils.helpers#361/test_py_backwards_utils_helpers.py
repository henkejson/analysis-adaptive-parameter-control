# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import py_backwards.utils.helpers as module_0


def test_case_0():
    none_type_0 = None
    none_type_1 = module_0.debug(none_type_0)


def test_case_1():
    variables_generator_0 = module_0.VariablesGenerator()


def test_case_2():
    int_0 = -461
    callable_0 = module_0.eager(int_0)
    none_type_0 = module_0.debug(callable_0)
    list_0 = [int_0, callable_0]
    none_type_1 = module_0.debug(callable_0)
    module_0.VariablesGenerator(*list_0)


def test_case_3():
    variables_generator_0 = module_0.VariablesGenerator()
    module_0.get_source(variables_generator_0)


def test_case_4():
    int_0 = -461
    none_type_0 = module_0.warn(int_0)


def test_case_5():
    int_0 = -460
    callable_0 = module_0.eager(int_0)
    none_type_0 = None
    callable_0.__call__(none_type_0, none_type_0, start=int_0)
