# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    auth_plugin_0 = module_0.AuthPlugin()


def test_case_1():
    auth_plugin_0 = module_0.AuthPlugin()
    auth_plugin_0.get_auth(auth_plugin_0)


def test_case_2():
    transport_plugin_0 = module_0.TransportPlugin()
    transport_plugin_0.get_adapter()


def test_case_3():
    str_0 = "."
    converter_plugin_0 = module_0.ConverterPlugin(str_0)


def test_case_4():
    module_0.FormatterPlugin()


def test_case_5():
    str_0 = "vXL!"
    converter_plugin_0 = module_0.ConverterPlugin(str_0)
    str_1 = "<{n!Oh"
    converter_plugin_0.convert(str_1)
