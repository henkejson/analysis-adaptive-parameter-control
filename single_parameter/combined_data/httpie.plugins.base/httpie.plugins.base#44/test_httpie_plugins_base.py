# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    module_0.FormatterPlugin()


def test_case_1():
    none_type_0 = None
    auth_plugin_0 = module_0.AuthPlugin()
    auth_plugin_0.get_auth(password=none_type_0)


def test_case_2():
    dict_0 = {}
    transport_plugin_0 = module_0.TransportPlugin(**dict_0)
    transport_plugin_0.get_adapter()


def test_case_3():
    str_0 = "mCV\n+-an;K"
    converter_plugin_0 = module_0.ConverterPlugin(str_0)
    converter_plugin_0.convert(str_0)
