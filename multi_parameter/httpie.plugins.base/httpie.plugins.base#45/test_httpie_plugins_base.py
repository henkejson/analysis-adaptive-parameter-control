# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    auth_plugin_0 = module_0.AuthPlugin()


def test_case_1():
    auth_plugin_0 = module_0.AuthPlugin()
    auth_plugin_0.get_auth()


def test_case_2():
    transport_plugin_0 = module_0.TransportPlugin()
    auth_plugin_0 = module_0.AuthPlugin()
    transport_plugin_0.get_adapter()


def test_case_3():
    str_0 = "QR>\rZ}"
    converter_plugin_0 = module_0.ConverterPlugin(str_0)


def test_case_4():
    str_0 = "f"
    converter_plugin_0 = module_0.ConverterPlugin(str_0)
    bytes_0 = b"\xea[\x0bF\x91"
    converter_plugin_0.convert(bytes_0)


def test_case_5():
    module_0.FormatterPlugin()
