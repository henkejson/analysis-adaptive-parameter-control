# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    module_0.FormatterPlugin()


def test_case_1():
    auth_plugin_0 = module_0.AuthPlugin()
    auth_plugin_0.get_auth()


def test_case_2():
    transport_plugin_0 = module_0.TransportPlugin()
    transport_plugin_0.get_adapter()


def test_case_3():
    str_0 = "En"
    str_1 = "%I\"J\x0c7B'_=_R^_7@akb"
    str_2 = "A@2"
    str_3 = "format_options"
    dict_0 = {str_1: str_1, str_2: str_2, str_3: str_3}
    formatter_plugin_0 = module_0.FormatterPlugin(**dict_0)
    str_4 = formatter_plugin_0.format_headers(str_0)
    str_5 = formatter_plugin_0.format_body(str_2, str_4)
    none_type_0 = None
    str_6 = ""
    converter_plugin_0 = module_0.ConverterPlugin(str_6)
    converter_plugin_0.convert(none_type_0)


def test_case_4():
    str_0 = "En"
    str_1 = "%I\"J\x0c7B'_=_R^_7@akb"
    str_2 = "A@2"
    str_3 = "format_options"
    dict_0 = {str_1: str_1, str_2: str_2, str_3: str_3}
    formatter_plugin_0 = module_0.FormatterPlugin(**dict_0)
    str_4 = formatter_plugin_0.format_headers(str_0)
    str_5 = formatter_plugin_0.format_body(str_2, str_4)
    none_type_0 = None
    str_6 = ""
    str_7 = formatter_plugin_0.format_metadata(dict_0)
    converter_plugin_0 = module_0.ConverterPlugin(str_6)
    converter_plugin_0.convert(none_type_0)
