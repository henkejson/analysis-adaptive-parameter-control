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
    str_0 = "_1GF%4pNh\tA=1kQ][I4"
    converter_plugin_0 = module_0.ConverterPlugin(str_0)
    transport_plugin_0.get_adapter()


def test_case_3():
    bytes_0 = b"\xf1\x96\xb6\x05\xa59]\xf6\x04\x03Y"
    dict_0 = {}
    converter_plugin_0 = module_0.ConverterPlugin(dict_0)
    converter_plugin_0.convert(bytes_0)


def test_case_4():
    module_0.FormatterPlugin()
