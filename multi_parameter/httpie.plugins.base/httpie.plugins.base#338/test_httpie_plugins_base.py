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
    auth_plugin_0 = module_0.AuthPlugin()
    converter_plugin_0 = module_0.ConverterPlugin(auth_plugin_0)
    bytes_0 = b"\xe4\xaf\x07\xcfG\xe8\x8f\xf7cB\x91\xdf=\xc3%hu"
    converter_plugin_0.convert(bytes_0)


def test_case_4():
    module_0.FormatterPlugin()
