# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    module_0.FormatterPlugin()


def test_case_1():
    bytes_0 = b"\x0f\xc2\xad\xa6"
    auth_plugin_0 = module_0.AuthPlugin()
    auth_plugin_0.get_auth(password=bytes_0)


def test_case_2():
    base_plugin_0 = module_0.BasePlugin()
    transport_plugin_0 = module_0.TransportPlugin()
    transport_plugin_0.get_adapter()


def test_case_3():
    bytes_0 = b"\n"
    transport_plugin_0 = module_0.TransportPlugin()
    str_0 = "*oL-79"
    converter_plugin_0 = module_0.ConverterPlugin(str_0)
    converter_plugin_0.convert(bytes_0)
