# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import flutils.decorators as module_0


def test_case_0():
    bytes_0 = b"\x1d9\xe4\x9f*v\r\x1a\xbb8\xe1SZ\x08\x17\x10\xe2"
    cached_property_0 = module_0.cached_property(bytes_0)
    none_type_0 = None
    var_0 = cached_property_0.__get__(none_type_0, none_type_0)
    var_0.__get__(bytes_0, cached_property_0)


def test_case_1():
    none_type_0 = None
    cached_property_0 = module_0.cached_property(none_type_0)
    cached_property_0.__get__(cached_property_0, cached_property_0)


def test_case_2():
    bytes_0 = b'#\x9c\xb4\xbc\x15U"^F\xed\xb7\xac<~\x8e\xa3V'
    cached_property_0 = module_0.cached_property(bytes_0)
