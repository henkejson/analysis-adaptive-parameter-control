# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import py_backwards.utils.helpers as module_0


def test_case_0():
    none_type_0 = None
    none_type_1 = module_0.debug(none_type_0)


def test_case_1():
    variables_generator_0 = module_0.VariablesGenerator()


def test_case_2():
    bool_0 = True
    callable_0 = module_0.eager(bool_0)
    str_0 = "1:/\x0b?-"
    callable_1 = module_0.eager(str_0)
    none_type_0 = module_0.warn(callable_0)
    none_type_1 = module_0.debug(none_type_0)
    none_type_2 = module_0.debug(callable_0)


def test_case_3():
    none_type_0 = None
    none_type_1 = module_0.debug(none_type_0)
    module_0.get_source(none_type_1)


def test_case_4():
    bool_0 = True
    callable_0 = module_0.eager(bool_0)
    str_0 = "1:/\x0b?-"
    callable_1 = module_0.eager(str_0)
    none_type_0 = module_0.warn(callable_0)
    callable_0.__call__(none_type_0, callable_0, qualname=none_type_0)
