# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    module_0.FormatterPlugin()


def test_case_1():
    auth_plugin_0 = module_0.AuthPlugin()
    auth_plugin_0.get_auth()


def test_case_2():
    list_0 = []
    transport_plugin_0 = module_0.TransportPlugin(*list_0)
    transport_plugin_0.get_adapter()


def test_case_3():
    int_0 = 3021
    list_0 = [int_0]
    converter_plugin_0 = module_0.ConverterPlugin(int_0)
    module_0.BasePlugin(*list_0)


def test_case_4():
    str_0 = "swF)&IqOAR$nE#Z"
    tuple_0 = (str_0, str_0)
    dict_0 = {}
    converter_plugin_0 = module_0.ConverterPlugin(dict_0)
    converter_plugin_0.convert(tuple_0)
