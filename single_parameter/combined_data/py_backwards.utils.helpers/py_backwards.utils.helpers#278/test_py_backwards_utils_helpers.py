# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import py_backwards.utils.helpers as module_0
import re as module_1


def test_case_0():
    int_0 = -18
    none_type_0 = module_0.debug(int_0)


def test_case_1():
    variables_generator_0 = module_0.VariablesGenerator()


def test_case_2():
    str_0 = "holder"
    none_type_0 = None
    none_type_1 = module_0.warn(none_type_0)
    none_type_2 = module_0.warn(str_0)
    callable_0 = module_0.eager(str_0)
    callable_1 = module_0.eager(str_0)
    none_type_3 = module_0.warn(str_0)


def test_case_3():
    dict_0 = {}
    module_0.get_source(dict_0)


def test_case_4():
    none_type_0 = None
    none_type_1 = module_0.warn(none_type_0)


def test_case_5():
    list_0 = []
    variables_generator_0 = module_0.VariablesGenerator(*list_0)
    str_0 = "UxM8,,vB\x0cs6W!"
    none_type_0 = module_0.warn(str_0)
    bytes_0 = b"\x1e\xf0\xbb&\xc5|T9\xae%<\x8bM\x93\x08\xfe-h\xfe"
    str_1 = 'v"yQ8o2O'
    none_type_1 = module_0.debug(str_1)
    callable_0 = module_0.eager(none_type_1)
    callable_1 = module_0.eager(callable_0)
    module_1.sub(bytes_0, callable_1, bytes_0)
