# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    transport_plugin_0 = module_0.TransportPlugin()


def test_case_1():
    auth_plugin_0 = module_0.AuthPlugin()
    auth_plugin_0.get_auth()


def test_case_2():
    base_plugin_0 = module_0.BasePlugin()
    base_plugin_1 = module_0.BasePlugin()
    transport_plugin_0 = module_0.TransportPlugin()
    transport_plugin_0.get_adapter()


def test_case_3():
    none_type_0 = None
    converter_plugin_0 = module_0.ConverterPlugin(none_type_0)
    module_0.FormatterPlugin()


def test_case_4():
    bool_0 = True
    converter_plugin_0 = module_0.ConverterPlugin(bool_0)
    transport_plugin_0 = module_0.TransportPlugin()
    bytes_0 = b"\xa1\x82\xb5\xbfe\x05\xa9"
    converter_plugin_0.convert(bytes_0)


def test_case_5():
    module_0.FormatterPlugin()


def test_case_6():
    base_plugin_0 = module_0.BasePlugin()
    str_0 = 'q"PB trYN2!pofEeSv5'
    str_1 = "G5sT9"
    str_2 = "format_options"
    dict_0 = {str_1: str_1, str_2: str_2}
    formatter_plugin_0 = module_0.FormatterPlugin(**dict_0)
    str_3 = formatter_plugin_0.format_body(str_0, str_0)
    base_plugin_1 = module_0.BasePlugin()
    bytes_0 = b"\x1b["
    base_plugin_1.convert(bytes_0)


def test_case_7():
    base_plugin_0 = module_0.BasePlugin()
    str_0 = 'q"PB trYN2!pofEeSv5'
    str_1 = "G5sT9"
    str_2 = "format_options"
    dict_0 = {str_1: str_1, str_2: str_2}
    formatter_plugin_0 = module_0.FormatterPlugin(**dict_0)
    str_3 = formatter_plugin_0.format_headers(str_1)
    str_4 = formatter_plugin_0.format_body(str_0, str_0)
    base_plugin_1 = module_0.BasePlugin()
    bytes_0 = b"\x1b["
    base_plugin_1.convert(bytes_0)


def test_case_8():
    base_plugin_0 = module_0.BasePlugin()
    str_0 = 'q"PB trYN2!pofEeSv5'
    str_1 = "G5sT9"
    str_2 = "format_options"
    dict_0 = {str_1: str_1, str_2: str_2}
    formatter_plugin_0 = module_0.FormatterPlugin(**dict_0)
    str_3 = formatter_plugin_0.format_body(str_0, str_0)
    str_4 = formatter_plugin_0.format_metadata(str_1)
    base_plugin_1 = module_0.BasePlugin()
    bytes_0 = b"\x1b["
    base_plugin_1.convert(bytes_0)
