# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    base_plugin_0 = module_0.BasePlugin()


def test_case_1():
    dict_0 = {}
    auth_plugin_0 = module_0.AuthPlugin(**dict_0)
    auth_plugin_0.get_auth()


def test_case_2():
    transport_plugin_0 = module_0.TransportPlugin()
    transport_plugin_0.get_adapter()


def test_case_3():
    none_type_0 = None
    converter_plugin_0 = module_0.ConverterPlugin(none_type_0)
    bool_0 = True
    list_0 = []
    auth_plugin_0 = module_0.AuthPlugin(*list_0)
    auth_plugin_0.get_auth(bool_0)


def test_case_4():
    float_0 = 590.9
    str_0 = "4'J|fg{aC"
    converter_plugin_0 = module_0.ConverterPlugin(str_0)
    converter_plugin_0.convert(float_0)


def test_case_5():
    module_0.FormatterPlugin()
