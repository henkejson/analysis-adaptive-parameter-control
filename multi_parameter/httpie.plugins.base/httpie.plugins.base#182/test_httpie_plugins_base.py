# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    transport_plugin_0 = module_0.TransportPlugin()


def test_case_1():
    auth_plugin_0 = module_0.AuthPlugin()
    auth_plugin_1 = module_0.AuthPlugin()
    auth_plugin_1.get_auth()


def test_case_2():
    transport_plugin_0 = module_0.TransportPlugin()
    transport_plugin_0.get_adapter()


def test_case_3():
    none_type_0 = None
    none_type_1 = None
    converter_plugin_0 = module_0.ConverterPlugin(none_type_1)
    converter_plugin_0.convert(none_type_0)


def test_case_4():
    module_0.FormatterPlugin()
