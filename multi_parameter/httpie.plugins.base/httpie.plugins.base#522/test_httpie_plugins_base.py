# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    module_0.FormatterPlugin()


def test_case_1():
    auth_plugin_0 = module_0.AuthPlugin()
    auth_plugin_0.get_auth()


def test_case_2():
    str_0 = "<HQ79D"
    converter_plugin_0 = module_0.ConverterPlugin(str_0)
    transport_plugin_0 = module_0.TransportPlugin()
    transport_plugin_0.get_adapter()


def test_case_3():
    str_0 = "K*tvgHHTs"
    str_1 = "' response metadata\n\n    The default behaviour is '"
    tuple_0 = (str_1, str_0)
    int_0 = 448
    tuple_1 = (str_0, tuple_0, int_0)
    bool_0 = False
    converter_plugin_0 = module_0.ConverterPlugin(bool_0)
    converter_plugin_0.convert(tuple_1)
