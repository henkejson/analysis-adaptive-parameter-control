# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import py_backwards.utils.helpers as module_0


def test_case_0():
    bool_0 = True
    none_type_0 = module_0.debug(bool_0)


def test_case_1():
    variables_generator_0 = module_0.VariablesGenerator()


def test_case_2():
    str_0 = "xrange"
    callable_0 = module_0.eager(str_0)


def test_case_3():
    none_type_0 = None
    module_0.get_source(none_type_0)


def test_case_4():
    bool_0 = True
    none_type_0 = module_0.warn(bool_0)


def test_case_5():
    bool_0 = True
    none_type_0 = module_0.debug(bool_0)
    callable_0 = module_0.eager(bool_0)
    callable_0.__call__(none_type_0, none_type_0, type=none_type_0, start=none_type_0)
