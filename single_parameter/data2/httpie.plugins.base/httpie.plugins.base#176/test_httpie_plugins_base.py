# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    base_plugin_0 = module_0.BasePlugin()


def test_case_1():
    transport_plugin_0 = module_0.TransportPlugin()
    str_0 = ""
    auth_plugin_0 = module_0.AuthPlugin()
    auth_plugin_0.get_auth(password=str_0)


def test_case_2():
    transport_plugin_0 = module_0.TransportPlugin()
    transport_plugin_0.get_adapter()


def test_case_3():
    str_0 = "\nV7qmY=<`\nZJQ"
    var_0 = module_0.ConverterPlugin(str_0)


def test_case_4():
    str_0 = "T~D0ZbFqG\x0ba#DqV&"
    converter_plugin_0 = module_0.ConverterPlugin(str_0)
    bytes_0 = b"y\x98\xccf\xb3U.5R\xf6y\xfc"
    converter_plugin_0.convert(bytes_0)


def test_case_5():
    module_0.FormatterPlugin()
