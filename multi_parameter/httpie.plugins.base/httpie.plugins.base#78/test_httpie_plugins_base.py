# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    transport_plugin_0 = module_0.TransportPlugin()


def test_case_1():
    auth_plugin_0 = module_0.AuthPlugin()
    auth_plugin_0.get_auth(auth_plugin_0)


def test_case_2():
    transport_plugin_0 = module_0.TransportPlugin()
    transport_plugin_0.get_adapter()


def test_case_3():
    str_0 = "Invalid Content-Range format "
    converter_plugin_0 = module_0.ConverterPlugin(str_0)


def test_case_4():
    module_0.FormatterPlugin()


def test_case_5():
    transport_plugin_0 = module_0.TransportPlugin()
    var_0 = module_0.ConverterPlugin(transport_plugin_0)
    bytes_0 = b""
    var_0.convert(bytes_0)
