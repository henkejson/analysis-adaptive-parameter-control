# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import flutils.decorators as module_0


def test_case_0():
    none_type_0 = None
    bytes_0 = b"!\x8d\xf6\x9f\xef\xdaTZ$\xbb\x18"
    cached_property_0 = module_0.cached_property(bytes_0)
    var_0 = cached_property_0.__get__(none_type_0, none_type_0)


def test_case_1():
    complex_0 = 683.73564 + 609.54j
    cached_property_0 = module_0.cached_property(complex_0)
    cached_property_0.__get__(complex_0, cached_property_0)


def test_case_2():
    complex_0 = 683.73564 + 609.54j
    cached_property_0 = module_0.cached_property(complex_0)
