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
    float_0 = 3621.75556
    callable_0 = module_1.eager(float_0)


def test_case_3():
    str_0 = "vQw2Y'K,IN\x0bf~+vVF]"
    none_type_0 = module_1.warn(str_0)
    module_1.get_source(none_type_0)


def test_case_4():
    str_0 = "vQw2Y'K,IN\x0bf~+vVF]"
    none_type_0 = module_1.warn(str_0)


def test_case_5():
    str_0 = "D}jRQ\\r%\rK6{pqxM\t2|p"
    none_type_0 = module_1.warn(str_0)
    none_type_1 = module_1.warn(none_type_0)
    callable_0 = module_1.eager(str_0)
    int_0 = -1900
    none_type_2 = module_1.warn(int_0)
    callable_1 = module_1.eager(str_0)
    callable_2 = module_1.eager(callable_0)
    variables_generator_0 = module_1.VariablesGenerator()
    callable_1.__call__(
        int_0,
        none_type_0,
        module=none_type_0,
        qualname=callable_1,
        type=callable_1,
        start=callable_0,
    )
