# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    auth_plugin_0 = module_0.AuthPlugin()


def test_case_1():
    none_type_0 = None
    auth_plugin_0 = module_0.AuthPlugin()
    auth_plugin_0.get_auth(none_type_0)


def test_case_2():
    transport_plugin_0 = module_0.TransportPlugin()
    transport_plugin_0.get_adapter()


def test_case_3():
    bytes_0 = b"x\xd4,\xae\xb4R"
    int_0 = -2183
    converter_plugin_0 = module_0.ConverterPlugin(int_0)
    converter_plugin_0.convert(bytes_0)


def test_case_4():
    module_0.FormatterPlugin()
