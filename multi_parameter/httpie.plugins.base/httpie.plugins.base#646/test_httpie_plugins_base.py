# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    transport_plugin_0 = module_0.TransportPlugin()


def test_case_1():
    none_type_0 = None
    auth_plugin_0 = module_0.AuthPlugin()
    auth_plugin_0.get_auth(none_type_0)


def test_case_2():
    transport_plugin_0 = module_0.TransportPlugin()
    transport_plugin_0.get_adapter()


def test_case_3():
    str_0 = '"Rp\x0cl_a'
    converter_plugin_0 = module_0.ConverterPlugin(str_0)
    transport_plugin_0 = module_0.TransportPlugin()
    transport_plugin_0.get_adapter()


def test_case_4():
    none_type_0 = None
    converter_plugin_0 = module_0.ConverterPlugin(none_type_0)
    bytes_0 = b""
    converter_plugin_0.convert(bytes_0)


def test_case_5():
    module_0.FormatterPlugin()


def test_case_6():
    str_0 = '-(Oja"oxXI'
    str_1 = "format_options"
    dict_0 = {str_1: str_0}
    formatter_plugin_0 = module_0.FormatterPlugin(**dict_0)
    str_2 = formatter_plugin_0.format_metadata(str_0)
    str_3 = formatter_plugin_0.format_body(str_0, str_0)
    none_type_0 = None
    list_0 = [none_type_0, none_type_0]
    module_0.TransportPlugin(*list_0)


def test_case_7():
    str_0 = '-(Oja"oxXI'
    str_1 = "format_options"
    dict_0 = {str_1: str_0}
    formatter_plugin_0 = module_0.FormatterPlugin(**dict_0)
    str_2 = formatter_plugin_0.format_metadata(str_0)
    str_3 = formatter_plugin_0.format_body(str_0, str_0)
    str_4 = formatter_plugin_0.format_headers(str_1)
    none_type_0 = None
    list_0 = [none_type_0, none_type_0]
    module_0.TransportPlugin(*list_0)
