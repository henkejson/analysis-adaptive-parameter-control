# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    base_plugin_0 = module_0.BasePlugin()


def test_case_1():
    none_type_0 = None
    dict_0 = {}
    auth_plugin_0 = module_0.AuthPlugin(**dict_0)
    auth_plugin_0.get_auth(password=none_type_0)


def test_case_2():
    transport_plugin_0 = module_0.TransportPlugin()
    transport_plugin_0.get_adapter()


def test_case_3():
    none_type_0 = None
    converter_plugin_0 = module_0.ConverterPlugin(none_type_0)


def test_case_4():
    str_0 = '*7q`a{*h8\r^)4b[\\SZ"^'
    converter_plugin_0 = module_0.ConverterPlugin(str_0)
    bytes_0 = b"\xf9b\xa1\xc4G\xabq\x9f\xd5\xecx\xc1\x92\x15\xac"
    converter_plugin_0.convert(bytes_0)


def test_case_5():
    module_0.FormatterPlugin()
