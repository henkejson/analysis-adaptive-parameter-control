# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import py_backwards.utils.helpers as module_0


def test_case_0():
    bool_0 = False
    none_type_0 = module_0.debug(bool_0)


def test_case_1():
    variables_generator_0 = module_0.VariablesGenerator()


def test_case_2():
    bytes_0 = b""
    callable_0 = module_0.eager(bytes_0)


def test_case_3():
    variables_generator_0 = module_0.VariablesGenerator()
    variables_generator_1 = module_0.VariablesGenerator()
    str_0 = ""
    none_type_0 = module_0.warn(str_0)
    module_0.get_source(variables_generator_0)


def test_case_4():
    str_0 = "N0{2bH"
    bool_0 = False
    none_type_0 = module_0.debug(bool_0)
    none_type_1 = module_0.warn(str_0)


def test_case_5():
    bool_0 = False
    variables_generator_0 = module_0.VariablesGenerator()
    callable_0 = module_0.eager(bool_0)
    none_type_0 = None
    callable_1 = module_0.eager(none_type_0)
    none_type_1 = None
    none_type_2 = module_0.debug(callable_0)
    callable_2 = module_0.eager(callable_0)
    callable_3 = module_0.eager(none_type_1)
    variables_generator_1 = module_0.VariablesGenerator()
    callable_4 = module_0.eager(none_type_1)
    none_type_3 = module_0.debug(variables_generator_1)
    callable_0.__call__(none_type_0, callable_3, type=none_type_3, start=none_type_0)
