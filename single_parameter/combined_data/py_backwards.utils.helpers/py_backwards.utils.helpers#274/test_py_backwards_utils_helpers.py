# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import py_backwards.utils.helpers as module_0
import re as module_1


def test_case_0():
    str_0 = "}1*g!J{}sVQIo+"
    none_type_0 = module_0.debug(str_0)


def test_case_1():
    variables_generator_0 = module_0.VariablesGenerator()


def test_case_2():
    none_type_0 = None
    callable_0 = module_0.eager(none_type_0)
    str_0 = "r$\rJ\nos\x0bc+XD#.\t"
    none_type_1 = module_0.warn(str_0)
    module_1.search(callable_0, none_type_1)


def test_case_3():
    str_0 = 'HEx\r\x0biv"]-'
    module_0.get_source(str_0)


def test_case_4():
    str_0 = "pipes"
    none_type_0 = module_0.warn(str_0)


def test_case_5():
    none_type_0 = None
    none_type_1 = module_0.debug(none_type_0)
    none_type_2 = None
    none_type_3 = module_0.debug(none_type_2)
    none_type_4 = module_0.warn(none_type_3)
    none_type_5 = module_0.warn(none_type_2)
    callable_0 = module_0.eager(none_type_2)
    callable_0.__call__(
        none_type_3, callable_0, module=none_type_0, type=none_type_3, start=none_type_4
    )
