# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import flutils.decorators as module_0
import builtins as module_1


def test_case_0():
    str_0 = "5(6#\n72G/:^rz\\\\OLQ|d"
    none_type_0 = None
    bool_0 = True
    cached_property_0 = module_0.cached_property(bool_0)
    var_0 = cached_property_0.__get__(none_type_0, none_type_0)
    var_0.__get__(str_0, str_0)


def test_case_1():
    object_0 = module_1.object()
    cached_property_0 = module_0.cached_property(object_0)
    cached_property_0.__get__(cached_property_0, cached_property_0)


def test_case_2():
    object_0 = module_1.object()
    cached_property_0 = module_0.cached_property(object_0)
