# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    module_0.FormatterPlugin()


def test_case_1():
    bytes_0 = b"\n\n"
    auth_plugin_0 = module_0.AuthPlugin()
    auth_plugin_0.get_auth(bytes_0, bytes_0)


def test_case_2():
    transport_plugin_0 = module_0.TransportPlugin()
    transport_plugin_0.get_adapter()


def test_case_3():
    str_0 = "Show version and exit."
    str_1 = "9Cy;"
    tuple_0 = (str_0, str_1)
    set_0 = {tuple_0, str_1, str_0}
    converter_plugin_0 = module_0.ConverterPlugin(set_0)
    module_0.TransportPlugin(*converter_plugin_0)


def test_case_4():
    bytes_0 = b"\xb0\xc4\xa7\xd2\x07\x07\xa6\xf2"
    converter_plugin_0 = module_0.ConverterPlugin(bytes_0)
    bytes_1 = b""
    converter_plugin_0.convert(bytes_1)
