# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import flutils.decorators as module_0


def test_case_0():
    bytes_0 = b"m&\xdb\xb6o+u|\xfa"
    cached_property_0 = module_0.cached_property(bytes_0)
    none_type_0 = None
    var_0 = cached_property_0.__get__(none_type_0, none_type_0)
    cached_property_0.__get__(bytes_0, bytes_0)


def test_case_1():
    set_0 = set()
    cached_property_0 = module_0.cached_property(set_0)
    cached_property_0.__get__(cached_property_0, set_0)


def test_case_2():
    bytes_0 = b"\xa0j\x1a\xd3LN\xe1\xb2\x05GI"
    cached_property_0 = module_0.cached_property(bytes_0)
