# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    module_0.FormatterPlugin()


def test_case_1():
    str_0 = "Fua+A;glHS\\P-"
    dict_0 = {}
    auth_plugin_0 = module_0.AuthPlugin(**dict_0)
    auth_plugin_0.get_auth(password=str_0)


def test_case_2():
    transport_plugin_0 = module_0.TransportPlugin()
    transport_plugin_0.get_adapter()


def test_case_3():
    str_0 = "\n        Decorator that converts a method with a single self argument into a\n        property cached on the instance.\n\n        A cached property can be made out of an existing method:\n        (e.g. ``url = cached_property(get_absolute_url)``).\n        The optional ``name`` argument is obsolete as of Python 3.6 and will be\n        deprecated in Django 4.0 (#30127).\n        "
    converter_plugin_0 = module_0.ConverterPlugin(str_0)
    str_1 = "( ,U]S_z_"
    dict_0 = {str_1: str_1}
    module_0.FormatterPlugin(**dict_0)


def test_case_4():
    str_0 = "#CCCC3D"
    converter_plugin_0 = module_0.ConverterPlugin(str_0)
    converter_plugin_1 = module_0.ConverterPlugin(str_0)
    bytes_0 = b"1y\x08\x85"
    converter_plugin_1.convert(bytes_0)
