# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    module_0.FormatterPlugin()


def test_case_1():
    auth_plugin_0 = module_0.AuthPlugin()
    auth_plugin_0.get_auth(auth_plugin_0)


def test_case_2():
    transport_plugin_0 = module_0.TransportPlugin()
    transport_plugin_0.get_adapter()


def test_case_3():
    str_0 = "{=4y.#B5&jI$w@*sI"
    converter_plugin_0 = module_0.ConverterPlugin(str_0)
    base_plugin_0 = module_0.BasePlugin()
    auth_plugin_0 = module_0.AuthPlugin()
    auth_plugin_0.get_auth(str_0)


def test_case_4():
    str_0 = "*g<|; ("
    bytes_0 = b"\x04\x07A&\x9c\x0by\x7fX\xa3\x1f\xebe>"
    converter_plugin_0 = module_0.ConverterPlugin(str_0)
    converter_plugin_0.convert(bytes_0)
