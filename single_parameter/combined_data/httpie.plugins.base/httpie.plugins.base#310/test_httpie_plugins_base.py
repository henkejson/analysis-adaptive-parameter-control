# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    module_0.FormatterPlugin()


def test_case_1():
    auth_plugin_0 = module_0.AuthPlugin()
    auth_plugin_0.get_auth()


def test_case_2():
    transport_plugin_0 = module_0.TransportPlugin()
    transport_plugin_0.get_adapter()


def test_case_3():
    bytes_0 = b"Z\x87\x10\x1b\x98V\xd5\xdf\x08\xa3"
    auth_plugin_0 = module_0.AuthPlugin()
    converter_plugin_0 = module_0.ConverterPlugin(auth_plugin_0)
    converter_plugin_0.convert(bytes_0)
