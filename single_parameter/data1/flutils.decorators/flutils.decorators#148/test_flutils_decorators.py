# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import flutils.decorators as module_0


def test_case_0():
    int_0 = 442
    cached_property_0 = module_0.cached_property(int_0)
    none_type_0 = None
    var_0 = cached_property_0.__get__(none_type_0, none_type_0)
    var_0.__get__(cached_property_0, cached_property_0)


def test_case_1():
    bool_0 = False
    cached_property_0 = module_0.cached_property(bool_0)
    cached_property_0.__get__(bool_0, bool_0)


def test_case_2():
    int_0 = 480
    cached_property_0 = module_0.cached_property(int_0)
