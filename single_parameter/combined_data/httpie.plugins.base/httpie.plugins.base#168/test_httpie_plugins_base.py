# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    transport_plugin_0 = module_0.TransportPlugin()


def test_case_1():
    transport_plugin_0 = module_0.TransportPlugin()
    converter_plugin_0 = module_0.ConverterPlugin(transport_plugin_0)
    none_type_0 = None
    auth_plugin_0 = module_0.AuthPlugin()
    auth_plugin_0.get_auth(password=none_type_0)


def test_case_2():
    transport_plugin_0 = module_0.TransportPlugin()
    transport_plugin_0.get_adapter()


def test_case_3():
    list_0 = []
    converter_plugin_0 = module_0.ConverterPlugin(list_0)


def test_case_4():
    str_0 = "/? "
    converter_plugin_0 = module_0.ConverterPlugin(str_0)
    auth_plugin_0 = module_0.AuthPlugin()
    bytes_0 = b"z\x16\xb1\xd8\x8a\x99\xde\x1c\x9ej)\x10|\xd9\xe8r\xd1\xaa\xb5"
    converter_plugin_0.convert(bytes_0)


def test_case_5():
    module_0.FormatterPlugin()
