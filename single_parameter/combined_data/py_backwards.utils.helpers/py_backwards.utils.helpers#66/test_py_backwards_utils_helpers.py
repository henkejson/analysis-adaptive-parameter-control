# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import py_backwards.utils.helpers as module_0


def test_case_0():
    str_0 = "/;a~i$Yp17y|^;Ld\x0b"
    none_type_0 = module_0.debug(str_0)


def test_case_1():
    variables_generator_0 = module_0.VariablesGenerator()


def test_case_2():
    variables_generator_0 = module_0.VariablesGenerator()
    none_type_0 = None
    callable_0 = module_0.eager(none_type_0)


def test_case_3():
    str_0 = "/;a~i$Yp17y|^;Ld\x0b"
    none_type_0 = module_0.warn(str_0)
    none_type_1 = module_0.debug(str_0)
    none_type_2 = module_0.warn(str_0)
    callable_0 = module_0.eager(none_type_2)
    module_0.get_source(callable_0)


def test_case_4():
    list_0 = []
    none_type_0 = module_0.warn(list_0)


def test_case_5():
    complex_0 = 914 - 1598j
    callable_0 = module_0.eager(complex_0)
    none_type_0 = None
    none_type_1 = module_0.debug(callable_0)
    callable_0.__call__(
        none_type_0, none_type_1, module=none_type_0, qualname=none_type_1
    )
