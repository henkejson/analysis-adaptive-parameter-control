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
    auth_plugin_0 = module_0.AuthPlugin()
    var_0 = module_0.ConverterPlugin(auth_plugin_0)


def test_case_4():
    transport_plugin_0 = module_0.TransportPlugin()
    bytes_0 = b"\xd3\xe1/\xd85\xee)\xe0qf,\xad\x0c\xbfr\xb0\x0f"
    none_type_0 = None
    converter_plugin_0 = module_0.ConverterPlugin(none_type_0)
    converter_plugin_0.convert(bytes_0)
