# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import flutils.decorators as module_0


def test_case_0():
    bytes_0 = b"\xb0\xac"
    cached_property_0 = module_0.cached_property(bytes_0)
    bytes_1 = b"\x0e\xcf\xb4\xb5"
    none_type_0 = None
    var_0 = cached_property_0.__get__(none_type_0, none_type_0)
    cached_property_0.__get__(bytes_0, bytes_1)


def test_case_1():
    bool_0 = False
    cached_property_0 = module_0.cached_property(bool_0)
    cached_property_0.__get__(cached_property_0, bool_0)


def test_case_2():
    none_type_0 = None
    cached_property_0 = module_0.cached_property(none_type_0)
