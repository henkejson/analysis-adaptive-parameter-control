# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import py_backwards.utils.helpers as module_0
import re as module_1


def test_case_0():
    bool_0 = True
    none_type_0 = module_0.debug(bool_0)


def test_case_1():
    variables_generator_0 = module_0.VariablesGenerator()


def test_case_2():
    tuple_0 = ()
    none_type_0 = module_0.debug(tuple_0)
    callable_0 = module_0.eager(none_type_0)
    module_1.escape(callable_0)


def test_case_3():
    var_0 = module_1.purge()
    module_0.get_source(var_0)


def test_case_4():
    str_0 = "?"
    none_type_0 = module_0.warn(str_0)


def test_case_5():
    str_0 = "^/t* "
    none_type_0 = None
    none_type_1 = module_0.debug(none_type_0)
    none_type_2 = module_0.debug(none_type_0)
    none_type_3 = module_0.warn(str_0)
    callable_0 = module_0.eager(none_type_3)
    callable_0.__call__(str_0, none_type_3)
