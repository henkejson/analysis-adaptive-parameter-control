# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import py_backwards.utils.helpers as module_0


def test_case_0():
    none_type_0 = None
    none_type_1 = module_0.debug(none_type_0)


def test_case_1():
    variables_generator_0 = module_0.VariablesGenerator()


def test_case_2():
    str_0 = "\\z\x0cgWDDabC]^HL)iBVjY"
    variables_generator_0 = module_0.VariablesGenerator()
    none_type_0 = module_0.debug(variables_generator_0)
    callable_0 = module_0.eager(str_0)
    module_0.get_source(callable_0)


def test_case_3():
    str_0 = ""
    none_type_0 = module_0.warn(str_0)
    module_0.get_source(none_type_0)


def test_case_4():
    str_0 = "B)Yd'`1q/"
    none_type_0 = module_0.warn(str_0)
    str_1 = "xmlrpc.server"
    list_0 = [str_1, str_1]
    module_0.VariablesGenerator(*list_0)


def test_case_5():
    none_type_0 = None
    variables_generator_0 = module_0.VariablesGenerator()
    none_type_1 = module_0.debug(none_type_0)
    callable_0 = module_0.eager(none_type_1)
    str_0 = ".moves.urllib_parse"
    callable_0.__call__(
        none_type_0, none_type_0, module=str_0, type=str_0, start=variables_generator_0
    )
