# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import py_backwards.utils.helpers as module_0


def test_case_0():
    variables_generator_0 = module_0.VariablesGenerator()
    none_type_0 = module_0.debug(variables_generator_0)


def test_case_1():
    variables_generator_0 = module_0.VariablesGenerator()


def test_case_2():
    none_type_0 = None
    callable_0 = module_0.eager(none_type_0)


def test_case_3():
    str_0 = "D^Jwca=PPAI[t"
    callable_0 = module_0.eager(str_0)
    none_type_0 = module_0.debug(callable_0)
    callable_0.__call__(str_0, callable_0, type=str_0)


def test_case_4():
    none_type_0 = None
    none_type_1 = module_0.debug(none_type_0)
    variables_generator_0 = module_0.VariablesGenerator()
    module_0.get_source(variables_generator_0)


def test_case_5():
    str_0 = "R<'onl~*K\t> V6L:=v^"
    none_type_0 = module_0.warn(str_0)
