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
    bytes_0 = b"2O\xb5_\x92\xfa\xe2\x0fR!\xd2"
    var_0 = module_0.escape(bytes_0)
    callable_0 = module_1.eager(var_0)
    none_type_0 = module_1.warn(callable_0)
    none_type_1 = module_1.warn(callable_0)
    none_type_2 = module_1.warn(var_0)
    none_type_3 = module_1.debug(var_0)


def test_case_3():
    float_0 = -1331.7
    module_1.get_source(float_0)


def test_case_4():
    str_0 = "tkMessageBox"
    none_type_0 = module_1.warn(str_0)


def test_case_5():
    bytes_0 = b"2O\xb5_\x92\xfa\xe2\x0fR!\xd2"
    var_0 = module_0.escape(bytes_0)
    callable_0 = module_1.eager(var_0)
    none_type_0 = module_1.warn(callable_0)
    module_0.subn(bytes_0, callable_0, var_0)
