# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    module_0.FormatterPlugin()


def test_case_1():
    str_0 = "\x0c?a0''_g=hVPnwyKXAN"
    auth_plugin_0 = module_0.AuthPlugin()
    auth_plugin_0.get_auth(str_0)


def test_case_2():
    transport_plugin_0 = module_0.TransportPlugin()
    transport_plugin_0.get_adapter()


def test_case_3():
    bool_0 = False
    str_0 = "Mq"
    str_1 = "S\r7!20_l)s]~?'|JW$"
    tuple_0 = (str_0, str_1)
    converter_plugin_0 = module_0.ConverterPlugin(tuple_0)
    converter_plugin_0.convert(bool_0)
