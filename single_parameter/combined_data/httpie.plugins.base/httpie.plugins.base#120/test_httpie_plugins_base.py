# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    transport_plugin_0 = module_0.TransportPlugin()


def test_case_1():
    str_0 = "f_B>-8Q"
    converter_plugin_0 = module_0.ConverterPlugin(str_0)
    none_type_0 = None
    str_1 = "User-Agent"
    converter_plugin_1 = module_0.ConverterPlugin(str_1)
    auth_plugin_0 = module_0.AuthPlugin()
    auth_plugin_0.get_auth(none_type_0, none_type_0)


def test_case_2():
    transport_plugin_0 = module_0.TransportPlugin()
    transport_plugin_0.get_adapter()


def test_case_3():
    bool_0 = False
    transport_plugin_0 = module_0.TransportPlugin()
    var_0 = module_0.ConverterPlugin(bool_0)
    var_0.convert(bool_0)


def test_case_4():
    module_0.FormatterPlugin()
