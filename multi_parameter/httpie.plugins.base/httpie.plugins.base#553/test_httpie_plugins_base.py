# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    base_plugin_0 = module_0.BasePlugin()


def test_case_1():
    str_0 = "{method} {path}{query} HTTP/1.1"
    auth_plugin_0 = module_0.AuthPlugin()
    auth_plugin_0.get_auth(str_0)


def test_case_2():
    transport_plugin_0 = module_0.TransportPlugin()
    transport_plugin_0.get_adapter()


def test_case_3():
    str_0 = "5&# Lj2KUKx?,gz#^="
    converter_plugin_0 = module_0.ConverterPlugin(str_0)


def test_case_4():
    module_0.FormatterPlugin()


def test_case_5():
    bytes_0 = b"\x86g\xa9\xc0\xa3`S/"
    str_0 = ""
    converter_plugin_0 = module_0.ConverterPlugin(str_0)
    converter_plugin_0.convert(bytes_0)
