# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    base_plugin_0 = module_0.BasePlugin()


def test_case_1():
    base_plugin_0 = module_0.BasePlugin()
    auth_plugin_0 = module_0.AuthPlugin()
    auth_plugin_0.get_auth()


def test_case_2():
    transport_plugin_0 = module_0.TransportPlugin()
    transport_plugin_0.get_adapter()


def test_case_3():
    str_0 = "#307842"
    converter_plugin_0 = module_0.ConverterPlugin(str_0)
    transport_plugin_0 = module_0.TransportPlugin()
    transport_plugin_0.get_adapter()


def test_case_4():
    str_0 = "\nCouldn’t resolve the given hostname. Please check the URL and try again."
    bool_0 = False
    converter_plugin_0 = module_0.ConverterPlugin(str_0)
    converter_plugin_0.convert(bool_0)


def test_case_5():
    module_0.FormatterPlugin()
