# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    module_0.FormatterPlugin()


def test_case_1():
    list_0 = []
    auth_plugin_0 = module_0.AuthPlugin(*list_0)
    auth_plugin_0.get_auth()


def test_case_2():
    transport_plugin_0 = module_0.TransportPlugin()
    transport_plugin_0.get_adapter()


def test_case_3():
    str_0 = "oHn*/GBU^+CRoO\t\x0bWS,C"
    converter_plugin_0 = module_0.ConverterPlugin(str_0)


def test_case_4():
    str_0 = "H]!f.\x0cSQ9n"
    str_1 = "Show any intermediary requests/responses."
    converter_plugin_0 = module_0.ConverterPlugin(str_1)
    converter_plugin_0.convert(str_0)
