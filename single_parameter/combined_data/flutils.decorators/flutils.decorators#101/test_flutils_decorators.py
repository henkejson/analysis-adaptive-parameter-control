# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import flutils.decorators as module_0


def test_case_0():
    none_type_0 = None
    none_type_1 = None
    cached_property_0 = module_0.cached_property(none_type_1)
    var_0 = cached_property_0.__get__(none_type_0, none_type_0)
    var_0.__get__(var_0, var_0)


def test_case_1():
    str_0 = "G+m\x0bg}."
    cached_property_0 = module_0.cached_property(str_0)
    cached_property_0.__get__(str_0, str_0)


def test_case_2():
    str_0 = "G+m\x0bg}."
    cached_property_0 = module_0.cached_property(str_0)
