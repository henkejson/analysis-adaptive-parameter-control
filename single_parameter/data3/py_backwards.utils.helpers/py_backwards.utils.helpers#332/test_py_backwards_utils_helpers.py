# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import py_backwards.utils.helpers as module_0


def test_case_0():
    none_type_0 = None
    none_type_1 = module_0.debug(none_type_0)


def test_case_1():
    variables_generator_0 = module_0.VariablesGenerator()


def test_case_2():
    none_type_0 = None
    callable_0 = module_0.eager(none_type_0)
    module_0.get_source(callable_0)


def test_case_3():
    bytes_0 = b"\xde\xb0C"
    module_0.get_source(bytes_0)


def test_case_4():
    str_0 = "/$U"
    none_type_0 = module_0.warn(str_0)


def test_case_5():
    variables_generator_0 = module_0.VariablesGenerator()
    none_type_0 = None
    none_type_1 = module_0.debug(none_type_0)
    none_type_2 = None
    callable_0 = module_0.eager(none_type_2)
    none_type_3 = module_0.warn(none_type_2)
    callable_1 = module_0.eager(callable_0)
    callable_1.__call__(callable_1, none_type_0, module=callable_1)
