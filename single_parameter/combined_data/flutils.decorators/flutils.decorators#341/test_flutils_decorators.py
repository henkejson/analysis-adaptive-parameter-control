# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import flutils.decorators as module_0


def test_case_0():
    bytes_0 = b"\x03R\xc5\xc6\\&H\xee\x7f\x11BN\xe4\xa8\xd6\xb0\x00\xfd\x91"
    none_type_0 = None
    int_0 = 448
    set_0 = {int_0, int_0, int_0}
    cached_property_0 = module_0.cached_property(set_0)
    var_0 = cached_property_0.__get__(none_type_0, set_0)
    var_0.__get__(bytes_0, bytes_0)


def test_case_1():
    none_type_0 = None
    cached_property_0 = module_0.cached_property(none_type_0)
    cached_property_0.__get__(cached_property_0, none_type_0)


def test_case_2():
    none_type_0 = None
    cached_property_0 = module_0.cached_property(none_type_0)
