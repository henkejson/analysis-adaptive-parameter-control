# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import py_backwards.utils.helpers as module_0


def test_case_0():
    str_0 = "$=hY\tTQ+hU_"
    none_type_0 = module_0.debug(str_0)


def test_case_1():
    variables_generator_0 = module_0.VariablesGenerator()


def test_case_2():
    list_0 = []
    variables_generator_0 = module_0.VariablesGenerator(*list_0)
    callable_0 = module_0.eager(list_0)


def test_case_3():
    bool_0 = True
    module_0.get_source(bool_0)


def test_case_4():
    str_0 = "}UT@w"
    none_type_0 = module_0.warn(str_0)


def test_case_5():
    none_type_0 = None
    none_type_1 = module_0.warn(none_type_0)
    callable_0 = module_0.eager(none_type_1)
    dict_0 = {}
    callable_0.__call__(none_type_0, dict_0, type=none_type_1, start=none_type_1)
