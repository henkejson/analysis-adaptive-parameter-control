# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    base_plugin_0 = module_0.BasePlugin()


def test_case_1():
    auth_plugin_0 = module_0.AuthPlugin()
    auth_plugin_0.get_auth()


def test_case_2():
    transport_plugin_0 = module_0.TransportPlugin()
    transport_plugin_0.get_adapter()


def test_case_3():
    transport_plugin_0 = module_0.TransportPlugin()
    converter_plugin_0 = module_0.ConverterPlugin(transport_plugin_0)


def test_case_4():
    bytes_0 = b"\xfe\xed\x14\xa8\xad\xb1\xdaC\xc7\x18\x86[\x0fM~\xd7\xe2\x91\xf2?"
    int_0 = -2598
    converter_plugin_0 = module_0.ConverterPlugin(int_0)
    converter_plugin_0.convert(bytes_0)


def test_case_5():
    module_0.FormatterPlugin()
