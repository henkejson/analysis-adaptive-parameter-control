# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    base_plugin_0 = module_0.BasePlugin()


def test_case_1():
    auth_plugin_0 = module_0.AuthPlugin()
    auth_plugin_0.get_auth()


def test_case_2():
    base_plugin_0 = module_0.BasePlugin()
    transport_plugin_0 = module_0.TransportPlugin()
    transport_plugin_0.get_adapter()


def test_case_3():
    float_0 = -233.0
    str_0 = "-_\roVw5Hu:9-+EbmlK"
    converter_plugin_0 = module_0.ConverterPlugin(str_0)
    converter_plugin_0.convert(float_0)


def test_case_4():
    module_0.FormatterPlugin()
