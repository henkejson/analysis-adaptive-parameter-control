# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import re as module_0
import py_backwards.utils.helpers as module_1
import collections as module_2


def test_case_0():
    variables_generator_0 = module_0.purge()
    none_type_0 = module_1.debug(variables_generator_0)


def test_case_1():
    variables_generator_0 = module_1.VariablesGenerator()


def test_case_2():
    variables_generator_0 = module_1.VariablesGenerator()
    ordered_dict_0 = module_2.OrderedDict()
    callable_0 = module_1.eager(ordered_dict_0)
    list_0 = [ordered_dict_0]
    module_1.VariablesGenerator(*list_0)


def test_case_3():
    bool_0 = False
    module_1.get_source(bool_0)


def test_case_4():
    str_0 = "1"
    none_type_0 = module_1.warn(str_0)


def test_case_5():
    bool_0 = True
    variables_generator_0 = module_1.VariablesGenerator()
    none_type_0 = module_1.debug(bool_0)
    callable_0 = module_1.eager(bool_0)
    str_0 = "os"
    none_type_1 = module_1.warn(str_0)
    callable_1 = module_1.eager(bool_0)
    callable_2 = module_1.eager(callable_1)
    callable_1.__call__(
        none_type_1, callable_0, none_type_1, start=variables_generator_0
    )
