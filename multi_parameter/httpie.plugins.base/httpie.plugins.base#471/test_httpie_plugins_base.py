# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    module_0.FormatterPlugin()


def test_case_1():
    str_0 = "(nrAQ'`KntiS:"
    auth_plugin_0 = module_0.AuthPlugin()
    auth_plugin_0.get_auth(password=str_0)


def test_case_2():
    transport_plugin_0 = module_0.TransportPlugin()
    transport_plugin_0.get_adapter()


def test_case_3():
    bool_0 = False
    converter_plugin_0 = module_0.ConverterPlugin(bool_0)


def test_case_4():
    transport_plugin_0 = module_0.TransportPlugin()
    converter_plugin_0 = module_0.ConverterPlugin(transport_plugin_0)
    converter_plugin_1 = module_0.ConverterPlugin(converter_plugin_0)
    str_0 = "-qk0;%xO"
    converter_plugin_0.convert(str_0)
