# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import re as module_0
import py_backwards.utils.helpers as module_1


def test_case_0():
    variables_generator_0 = module_0.purge()
    none_type_0 = module_1.debug(variables_generator_0)


def test_case_1():
    variables_generator_0 = module_1.VariablesGenerator()


def test_case_2():
    bytes_0 = b"]=:@-K\xa83\xfd\xf7I\x85I'"
    callable_0 = module_1.eager(bytes_0)


def test_case_3():
    bool_0 = False
    none_type_0 = module_1.debug(bool_0)
    module_1.get_source(none_type_0)


def test_case_4():
    str_0 = "v  \x0c }Sprj"
    none_type_0 = module_1.warn(str_0)


def test_case_5():
    str_0 = "arH=ju"
    callable_0 = module_1.eager(str_0)
    callable_1 = module_1.eager(str_0)
    none_type_0 = module_1.debug(callable_0)
    callable_0.__call__(none_type_0, none_type_0, none_type_0)
