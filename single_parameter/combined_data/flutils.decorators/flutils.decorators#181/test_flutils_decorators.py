# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import flutils.decorators as module_0


def test_case_0():
    bytes_0 = b"~\xbd\x91\x02:\xaa\xfb"
    cached_property_0 = module_0.cached_property(bytes_0)
    none_type_0 = None
    var_0 = cached_property_0.__get__(none_type_0, none_type_0)
    cached_property_0.__get__(bytes_0, bytes_0)


def test_case_1():
    bytes_0 = b"~\x13\x91\x02:\xaa\xa1"
    cached_property_0 = module_0.cached_property(bytes_0)
    cached_property_0.__get__(bytes_0, bytes_0)


def test_case_2():
    bytes_0 = b"~\x13\x91\x02:\xaa\xa1"
    cached_property_0 = module_0.cached_property(bytes_0)
