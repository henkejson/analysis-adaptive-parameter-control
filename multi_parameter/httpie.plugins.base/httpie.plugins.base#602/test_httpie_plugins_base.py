# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    module_0.FormatterPlugin()


def test_case_1():
    auth_plugin_0 = module_0.AuthPlugin()
    auth_plugin_0.get_auth(auth_plugin_0)


def test_case_2():
    list_0 = []
    transport_plugin_0 = module_0.TransportPlugin(*list_0)
    transport_plugin_0.get_adapter()


def test_case_3():
    auth_plugin_0 = module_0.AuthPlugin()
    converter_plugin_0 = module_0.ConverterPlugin(auth_plugin_0)


def test_case_4():
    bytes_0 = b"\t\x80sw\x19F\xf9<8\x10\xe3\xcdP\xff\x9b\x15"
    str_0 = "(default) Serialize data items from the command line as a JSON object."
    str_1 = "tG"
    tuple_0 = (str_0, str_1)
    converter_plugin_0 = module_0.ConverterPlugin(tuple_0)
    converter_plugin_0.convert(bytes_0)
