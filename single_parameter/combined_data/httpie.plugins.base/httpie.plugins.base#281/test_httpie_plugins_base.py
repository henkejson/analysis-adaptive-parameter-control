# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    transport_plugin_0 = module_0.TransportPlugin()


def test_case_1():
    transport_plugin_0 = module_0.TransportPlugin()
    auth_plugin_0 = module_0.AuthPlugin()
    auth_plugin_0.get_auth()


def test_case_2():
    transport_plugin_0 = module_0.TransportPlugin()
    transport_plugin_0.get_adapter()


def test_case_3():
    bytes_0 = b'\xea\x0e\xad\x9b"\xee\xda\x96\xc7\x91%'
    set_0 = {bytes_0}
    converter_plugin_0 = module_0.ConverterPlugin(set_0)


def test_case_4():
    none_type_0 = None
    none_type_1 = None
    converter_plugin_0 = module_0.ConverterPlugin(none_type_1)
    converter_plugin_0.convert(none_type_0)


def test_case_5():
    module_0.FormatterPlugin()
