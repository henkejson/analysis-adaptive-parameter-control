# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import py_backwards.utils.helpers as module_0
import re as module_1


def test_case_0():
    float_0 = 3000.49
    none_type_0 = module_0.debug(float_0)


def test_case_1():
    variables_generator_0 = module_0.VariablesGenerator()


def test_case_2():
    var_0 = module_1.purge()
    callable_0 = module_0.eager(var_0)


def test_case_3():
    none_type_0 = None
    module_0.get_source(none_type_0)


def test_case_4():
    str_0 = "email.MIMEBase"
    none_type_0 = module_0.warn(str_0)


def test_case_5():
    none_type_0 = None
    callable_0 = module_0.eager(none_type_0)
    callable_0.__call__(none_type_0, callable_0, type=callable_0)
