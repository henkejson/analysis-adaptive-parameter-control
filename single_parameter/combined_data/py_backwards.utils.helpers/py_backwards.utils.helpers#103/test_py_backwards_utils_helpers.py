# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import enum as module_0
import py_backwards.utils.helpers as module_1
import re as module_2


def test_case_0():
    var_0 = module_0._EnumDict()
    none_type_0 = module_1.debug(var_0)


def test_case_1():
    variables_generator_0 = module_1.VariablesGenerator()


def test_case_2():
    var_0 = module_2.purge()
    callable_0 = module_1.eager(var_0)


def test_case_3():
    str_0 = "u0s"
    module_1.get_source(str_0)


def test_case_4():
    str_0 = "V@"
    none_type_0 = module_1.warn(str_0)


def test_case_5():
    bytes_0 = b"\x96\xae!/J?\x9c+\x7f?T\rQ\xde."
    none_type_0 = module_1.debug(bytes_0)
    callable_0 = module_1.eager(bytes_0)
    callable_0.__call__(callable_0, bytes_0, module=callable_0, type=none_type_0)
