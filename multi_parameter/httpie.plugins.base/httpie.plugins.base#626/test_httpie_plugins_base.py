# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    module_0.FormatterPlugin()


def test_case_1():
    none_type_0 = None
    auth_plugin_0 = module_0.AuthPlugin()
    auth_plugin_0.get_auth(none_type_0)


def test_case_2():
    transport_plugin_0 = module_0.TransportPlugin()
    transport_plugin_0.get_adapter()


def test_case_3():
    str_0 = "r,U{`\nH'.o"
    converter_plugin_0 = module_0.ConverterPlugin(str_0)


def test_case_4():
    bytes_0 = b"\x9c\xc0c\xb9\xaa\xcc$\x1e\x11@\x0b\x1c\xe0"
    none_type_0 = None
    converter_plugin_0 = module_0.ConverterPlugin(none_type_0)
    converter_plugin_0.convert(bytes_0)


def test_case_5():
    str_0 = "r,U{`\nH'.oaS"
    str_1 = "USER[:PASS] | TOKEN"
    str_2 = "format_options"
    str_3 = "?~wP*N4|WC7j=kVt\t\t"
    dict_0 = {str_2: str_2, str_0: str_0, str_3: str_3}
    formatter_plugin_0 = module_0.FormatterPlugin(**dict_0)
    str_4 = formatter_plugin_0.format_headers(str_1)


def test_case_6():
    str_0 = "r,U{`\nH'.oaS"
    str_1 = "format_options"
    str_2 = "?~wP*N4|WC7j=kVt\t\t"
    dict_0 = {str_1: str_1, str_0: str_0, str_2: str_2}
    formatter_plugin_0 = module_0.FormatterPlugin(**dict_0)
    str_3 = formatter_plugin_0.format_metadata(str_0)
    auth_plugin_0 = module_0.AuthPlugin()
    auth_plugin_0.get_auth(dict_0)


def test_case_7():
    str_0 = "r,U{`\nH'.oaS"
    str_1 = "format_options"
    str_2 = "?~wP*N4|WC7j=kVt\t\t"
    dict_0 = {str_1: str_1, str_0: str_0, str_2: str_2}
    formatter_plugin_0 = module_0.FormatterPlugin(**dict_0)
    str_3 = formatter_plugin_0.format_body(str_2, str_2)
    transport_plugin_0 = module_0.TransportPlugin()
