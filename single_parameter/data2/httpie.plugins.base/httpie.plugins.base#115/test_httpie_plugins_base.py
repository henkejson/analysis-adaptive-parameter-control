# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    base_plugin_0 = module_0.BasePlugin()


def test_case_1():
    str_0 = "y`S#2|pleV0J\x0c\n"
    auth_plugin_0 = module_0.AuthPlugin()
    auth_plugin_0.get_auth(password=str_0)


def test_case_2():
    transport_plugin_0 = module_0.TransportPlugin()
    transport_plugin_0.get_adapter()


def test_case_3():
    str_0 = "#E3822B"
    converter_plugin_0 = module_0.ConverterPlugin(str_0)


def test_case_4():
    bytes_0 = b"\r\n\r\n"
    str_0 = "tW1cPq>4MFG_3\x0c"
    converter_plugin_0 = module_0.ConverterPlugin(str_0)
    converter_plugin_0.convert(bytes_0)


def test_case_5():
    module_0.FormatterPlugin()
