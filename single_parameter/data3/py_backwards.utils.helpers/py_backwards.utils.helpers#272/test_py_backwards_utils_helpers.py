# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import py_backwards.utils.helpers as module_0


def test_case_0():
    none_type_0 = None
    none_type_1 = module_0.debug(none_type_0)


def test_case_1():
    variables_generator_0 = module_0.VariablesGenerator()


def test_case_2():
    none_type_0 = None
    callable_0 = module_0.eager(none_type_0)


def test_case_3():
    bool_0 = False
    module_0.get_source(bool_0)


def test_case_4():
    int_0 = 709
    none_type_0 = module_0.warn(int_0)


def test_case_5():
    variables_generator_0 = module_0.VariablesGenerator()
    bool_0 = False
    callable_0 = module_0.eager(bool_0)
    none_type_0 = module_0.warn(callable_0)
    none_type_1 = module_0.debug(callable_0)
    none_type_2 = module_0.warn(bool_0)
    callable_1 = module_0.eager(callable_0)
    none_type_3 = module_0.debug(none_type_2)
    callable_2 = module_0.eager(none_type_3)
    none_type_4 = module_0.warn(none_type_2)
    callable_2.__call__(callable_2, variables_generator_0)
