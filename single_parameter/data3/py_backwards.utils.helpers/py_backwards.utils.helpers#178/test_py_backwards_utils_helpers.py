# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import py_backwards.utils.helpers as module_0


def test_case_0():
    bool_0 = True
    none_type_0 = module_0.debug(bool_0)


def test_case_1():
    variables_generator_0 = module_0.VariablesGenerator()


def test_case_2():
    float_0 = 1877.0
    dict_0 = {float_0: float_0}
    callable_0 = module_0.eager(dict_0)


def test_case_3():
    bool_0 = True
    none_type_0 = None
    none_type_1 = module_0.debug(none_type_0)
    module_0.get_source(bool_0)


def test_case_4():
    str_0 = "vv}"
    none_type_0 = module_0.warn(str_0)


def test_case_5():
    none_type_0 = None
    none_type_1 = module_0.warn(none_type_0)
    callable_0 = module_0.eager(none_type_0)
    callable_0.__call__(none_type_1, none_type_0, type=none_type_1)
