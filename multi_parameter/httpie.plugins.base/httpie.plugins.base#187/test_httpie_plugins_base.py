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
    none_type_0 = None
    converter_plugin_0 = module_0.ConverterPlugin(none_type_0)
    bool_0 = True
    auth_plugin_0 = module_0.AuthPlugin()
    auth_plugin_0.get_auth(bool_0, converter_plugin_0)


def test_case_4():
    str_0 = "LyoG,};Ue9^3-nxW"
    tuple_0 = (str_0, str_0)
    bytes_0 = b"\n\n"
    converter_plugin_0 = module_0.ConverterPlugin(tuple_0)
    converter_plugin_0.convert(bytes_0)
