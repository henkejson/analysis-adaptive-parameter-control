# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    transport_plugin_0 = module_0.TransportPlugin()


def test_case_1():
    auth_plugin_0 = module_0.AuthPlugin()
    auth_plugin_0.get_auth()


def test_case_2():
    transport_plugin_0 = module_0.TransportPlugin()
    transport_plugin_0.get_adapter()


def test_case_3():
    bytes_0 = b"\n\n"
    converter_plugin_0 = module_0.ConverterPlugin(bytes_0)
    transport_plugin_0 = module_0.TransportPlugin()
    transport_plugin_0.get_adapter()


def test_case_4():
    module_0.FormatterPlugin()


def test_case_5():
    bytes_0 = b"\n\n"
    converter_plugin_0 = module_0.ConverterPlugin(bytes_0)
    converter_plugin_0.convert(bytes_0)
