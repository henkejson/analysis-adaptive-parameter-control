# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    auth_plugin_0 = module_0.AuthPlugin()


def test_case_1():
    auth_plugin_0 = module_0.AuthPlugin()
    auth_plugin_0.get_auth()


def test_case_2():
    dict_0 = {}
    transport_plugin_0 = module_0.TransportPlugin(**dict_0)
    transport_plugin_0.get_adapter()


def test_case_3():
    auth_plugin_0 = module_0.AuthPlugin()
    str_0 = "{5WXkT\\}y_a\rkk_Aw"
    converter_plugin_0 = module_0.ConverterPlugin(str_0)
    list_0 = [converter_plugin_0]
    module_0.AuthPlugin(*list_0)


def test_case_4():
    bytes_0 = b"\xced\xebF3<\xc1\x9e\xfeZ8r"
    none_type_0 = None
    converter_plugin_0 = module_0.ConverterPlugin(none_type_0)
    converter_plugin_0.convert(bytes_0)


def test_case_5():
    module_0.FormatterPlugin()
