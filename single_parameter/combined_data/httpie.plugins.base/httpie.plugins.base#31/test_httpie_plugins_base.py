# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    module_0.FormatterPlugin()


def test_case_1():
    dict_0 = {}
    auth_plugin_0 = module_0.AuthPlugin(**dict_0)
    auth_plugin_0.get_auth()


def test_case_2():
    transport_plugin_0 = module_0.TransportPlugin()
    transport_plugin_0.get_adapter()


def test_case_3():
    bool_0 = False
    converter_plugin_0 = module_0.ConverterPlugin(bool_0)


def test_case_4():
    bool_0 = True
    bool_1 = False
    converter_plugin_0 = module_0.ConverterPlugin(bool_1)
    converter_plugin_0.convert(bool_0)
