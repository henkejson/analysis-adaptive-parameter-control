# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import flutils.decorators as module_0


def test_case_0():
    tuple_0 = ()
    cached_property_0 = module_0.cached_property(tuple_0)
    none_type_0 = None
    var_0 = cached_property_0.__get__(none_type_0, cached_property_0)
    cached_property_0.__get__(cached_property_0, cached_property_0)


def test_case_1():
    tuple_0 = ()
    cached_property_0 = module_0.cached_property(tuple_0)
    cached_property_0.__get__(cached_property_0, cached_property_0)


def test_case_2():
    complex_0 = 2430 - 3081j
    cached_property_0 = module_0.cached_property(complex_0)
