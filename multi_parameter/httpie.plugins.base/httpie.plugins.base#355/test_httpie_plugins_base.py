# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    base_plugin_0 = module_0.BasePlugin()


def test_case_1():
    float_0 = -2604.37934
    auth_plugin_0 = module_0.AuthPlugin()
    auth_plugin_0.get_auth(password=float_0)


def test_case_2():
    transport_plugin_0 = module_0.TransportPlugin()
    transport_plugin_0.get_adapter()


def test_case_3():
    none_type_0 = None
    str_0 = "q9}mXUi8+*s\\lQ3MS\n\n"
    converter_plugin_0 = module_0.ConverterPlugin(str_0)
    auth_plugin_0 = module_0.AuthPlugin()
    auth_plugin_0.get_auth(none_type_0)


def test_case_4():
    bool_0 = True
    converter_plugin_0 = module_0.ConverterPlugin(bool_0)
    none_type_0 = None
    converter_plugin_0.convert(none_type_0)


def test_case_5():
    module_0.FormatterPlugin()
