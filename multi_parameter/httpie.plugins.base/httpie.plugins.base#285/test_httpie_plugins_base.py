# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    module_0.FormatterPlugin()


def test_case_1():
    str_0 = "white"
    list_0 = []
    auth_plugin_0 = module_0.AuthPlugin(*list_0)
    auth_plugin_0.get_auth(str_0)


def test_case_2():
    list_0 = []
    transport_plugin_0 = module_0.TransportPlugin(*list_0)
    transport_plugin_0.get_adapter()


def test_case_3():
    none_type_0 = None
    converter_plugin_0 = module_0.ConverterPlugin(none_type_0)


def test_case_4():
    str_0 = "R$I\t(dnU7;}_1/("
    str_1 = "PC_NAME_MAX"
    converter_plugin_0 = module_0.ConverterPlugin(str_1)
    converter_plugin_0.convert(str_0)
