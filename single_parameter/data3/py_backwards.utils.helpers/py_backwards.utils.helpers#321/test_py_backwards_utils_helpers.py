# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import py_backwards.utils.helpers as module_0
import builtins as module_1


def test_case_0():
    str_0 = "GF_|TKH\n6^4o"
    none_type_0 = module_0.debug(str_0)


def test_case_1():
    variables_generator_0 = module_0.VariablesGenerator()


def test_case_2():
    str_0 = "GF_|TKH\n6^4o"
    callable_0 = module_0.eager(str_0)
    none_type_0 = module_0.debug(str_0)


def test_case_3():
    exception_0 = module_1.Exception()
    module_0.get_source(exception_0)


def test_case_4():
    str_0 = "M`5x+o\n"
    none_type_0 = module_0.warn(str_0)


def test_case_5():
    bool_0 = True
    callable_0 = module_0.eager(bool_0)
    exception_0 = module_1.Exception()
    callable_0.__call__(exception_0, exception_0, qualname=bool_0)
