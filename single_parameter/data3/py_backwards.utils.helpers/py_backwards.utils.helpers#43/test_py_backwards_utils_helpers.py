# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import py_backwards.utils.helpers as module_0


def test_case_0():
    list_0 = []
    none_type_0 = module_0.debug(list_0)


def test_case_1():
    variables_generator_0 = module_0.VariablesGenerator()


def test_case_2():
    none_type_0 = None
    callable_0 = module_0.eager(none_type_0)


def test_case_3():
    list_0 = []
    none_type_0 = module_0.debug(list_0)
    module_0.get_source(none_type_0)


def test_case_4():
    str_0 = "email.MIMEMultipart"
    none_type_0 = module_0.warn(str_0)


def test_case_5():
    bool_0 = True
    callable_0 = module_0.eager(bool_0)
    none_type_0 = module_0.warn(bool_0)
    callable_1 = module_0.eager(none_type_0)
    callable_0.__call__(bool_0, bool_0, none_type_0, start=bool_0)
