# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    auth_plugin_0 = module_0.AuthPlugin()


def test_case_1():
    auth_plugin_0 = module_0.AuthPlugin()
    auth_plugin_0.get_auth()


def test_case_2():
    dict_0 = {}
    transport_plugin_0 = module_0.TransportPlugin(**dict_0)
    transport_plugin_0.get_adapter()


def test_case_3():
    auth_plugin_0 = module_0.AuthPlugin()
    converter_plugin_0 = module_0.ConverterPlugin(auth_plugin_0)


def test_case_4():
    int_0 = -1579
    str_0 = ">Qs\rw*"
    converter_plugin_0 = module_0.ConverterPlugin(str_0)
    converter_plugin_0.convert(int_0)


def test_case_5():
    module_0.FormatterPlugin()
