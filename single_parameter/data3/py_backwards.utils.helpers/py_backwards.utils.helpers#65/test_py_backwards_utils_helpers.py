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
    float_0 = 393.32
    callable_0 = module_1.eager(float_0)


def test_case_3():
    none_type_0 = None
    module_1.get_source(none_type_0)


def test_case_4():
    str_0 = "4o(wpYc3RmMLQ<"
    none_type_0 = module_1.warn(str_0)


def test_case_5():
    str_0 = " mY:wP}ha~2'iE\\W{\nr"
    callable_0 = module_1.eager(str_0)
    none_type_0 = module_1.warn(str_0)
    callable_1 = module_1.eager(str_0)
    callable_2 = module_1.eager(callable_1)
    none_type_1 = module_1.warn(str_0)
    none_type_2 = module_1.debug(callable_1)
    callable_3 = module_1.eager(callable_0)
    callable_0.__call__(none_type_0, none_type_1)
