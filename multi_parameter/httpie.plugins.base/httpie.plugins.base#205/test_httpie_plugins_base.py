# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    base_plugin_0 = module_0.BasePlugin()


def test_case_1():
    auth_plugin_0 = module_0.AuthPlugin()
    transport_plugin_0 = module_0.TransportPlugin()
    converter_plugin_0 = module_0.ConverterPlugin(transport_plugin_0)
    converter_plugin_1 = module_0.ConverterPlugin(converter_plugin_0)
    none_type_0 = None
    auth_plugin_0.get_auth(none_type_0)


def test_case_2():
    transport_plugin_0 = module_0.TransportPlugin()
    transport_plugin_0.get_adapter()


def test_case_3():
    bytes_0 = b"\x00"
    str_0 = "K$7C\t(#%>>"
    converter_plugin_0 = module_0.ConverterPlugin(str_0)
    converter_plugin_0.convert(bytes_0)


def test_case_4():
    module_0.FormatterPlugin()
