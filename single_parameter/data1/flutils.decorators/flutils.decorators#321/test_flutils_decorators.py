# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import flutils.decorators as module_0


def test_case_0():
    str_0 = "__additional_attrs__ keys must be strings. in %r"
    cached_property_0 = module_0.cached_property(str_0)
    none_type_0 = None
    var_0 = cached_property_0.__get__(none_type_0, cached_property_0)


def test_case_1():
    int_0 = 223
    cached_property_0 = module_0.cached_property(int_0)
    cached_property_0.__get__(int_0, cached_property_0)


def test_case_2():
    int_0 = 223
    cached_property_0 = module_0.cached_property(int_0)
