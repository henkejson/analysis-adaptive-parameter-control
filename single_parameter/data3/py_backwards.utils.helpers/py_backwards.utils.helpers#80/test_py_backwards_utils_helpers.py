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
    none_type_0 = None
    none_type_1 = module_1.debug(none_type_0)
    callable_0 = module_1.eager(none_type_0)
    none_type_2 = module_1.warn(none_type_0)
    module_0.split(callable_0, none_type_0)


def test_case_3():
    str_0 = "AMqC}m8xV=562"
    none_type_0 = None
    none_type_1 = module_1.debug(none_type_0)
    callable_0 = module_1.eager(str_0)
    none_type_2 = module_1.debug(callable_0)
    module_1.get_source(callable_0)


def test_case_4():
    str_0 = "VTb/S*3 -(\x0c7en@][W"
    none_type_0 = module_1.warn(str_0)


def test_case_5():
    str_0 = "UP X,<KY6l.' [x<"
    none_type_0 = module_1.warn(str_0)
    float_0 = 1810.0
    callable_0 = module_1.eager(float_0)
    str_1 = ""
    none_type_1 = module_1.warn(str_1)
    none_type_2 = None
    callable_0.__call__(none_type_2, none_type_2)
