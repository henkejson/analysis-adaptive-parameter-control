# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import py_backwards.utils.helpers as module_0
import re as module_1


def test_case_0():
    str_0 = "UA@"
    none_type_0 = module_0.debug(str_0)


def test_case_1():
    variables_generator_0 = module_0.VariablesGenerator()


def test_case_2():
    bytes_0 = b"\x04:\xafkN"
    callable_0 = module_0.eager(bytes_0)
    module_0.get_source(callable_0)


def test_case_3():
    tuple_0 = ()
    none_type_0 = module_0.debug(tuple_0)
    str_0 = "UA@"
    none_type_1 = module_0.warn(str_0)
    none_type_2 = module_0.debug(none_type_0)
    module_1.template(tuple_0)


def test_case_4():
    str_0 = "tkinter.messagebox"
    none_type_0 = module_0.debug(str_0)
    none_type_1 = module_0.warn(str_0)
    variables_generator_0 = module_0.VariablesGenerator()
    callable_0 = module_0.eager(str_0)
    callable_0.__call__(callable_0, callable_0, qualname=variables_generator_0)
