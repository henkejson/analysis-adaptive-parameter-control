# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import flutils.decorators as module_0


def test_case_0():
    int_0 = -5862
    cached_property_0 = module_0.cached_property(int_0)
    none_type_0 = None
    var_0 = cached_property_0.__get__(none_type_0, none_type_0)
    cached_property_0.__get__(cached_property_0, int_0)


def test_case_1():
    int_0 = -5830
    cached_property_0 = module_0.cached_property(int_0)
    cached_property_0.__get__(cached_property_0, int_0)


def test_case_2():
    int_0 = -5830
    cached_property_0 = module_0.cached_property(int_0)
