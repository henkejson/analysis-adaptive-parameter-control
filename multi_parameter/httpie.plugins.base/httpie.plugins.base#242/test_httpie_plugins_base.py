# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    module_0.FormatterPlugin()


def test_case_1():
    list_0 = []
    auth_plugin_0 = module_0.AuthPlugin(*list_0)
    auth_plugin_0.get_auth()


def test_case_2():
    dict_0 = {}
    transport_plugin_0 = module_0.TransportPlugin(**dict_0)
    transport_plugin_0.get_adapter()


def test_case_3():
    bool_0 = False
    converter_plugin_0 = module_0.ConverterPlugin(bool_0)
    auth_plugin_0 = module_0.AuthPlugin()
    auth_plugin_0.get_auth()


def test_case_4():
    str_0 = "'s^zq9"
    converter_plugin_0 = module_0.ConverterPlugin(str_0)
    bytes_0 = b"T\x1f"
    converter_plugin_0.convert(bytes_0)
