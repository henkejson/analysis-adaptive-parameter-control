# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    auth_plugin_0 = module_0.AuthPlugin()


def test_case_1():
    auth_plugin_0 = module_0.AuthPlugin()
    auth_plugin_0.get_auth()


def test_case_2():
    transport_plugin_0 = module_0.TransportPlugin()
    transport_plugin_0.get_adapter()


def test_case_3():
    bytes_0 = b"\x07+\xcd\xbbN?\x83-\x8e\x9f]\x99\xcb\xcb/\xa6\x9b\x0e\\\xa1"
    str_0 = "'Xngoh-"
    converter_plugin_0 = module_0.ConverterPlugin(str_0)
    converter_plugin_0.convert(bytes_0)


def test_case_4():
    module_0.FormatterPlugin()


def test_case_5():
    float_0 = 2687.63
    str_0 = "format_options"
    none_type_0 = None
    str_1 = "+<"
    dict_0 = {
        str_0: none_type_0,
        str_1: none_type_0,
        str_0: none_type_0,
        str_0: none_type_0,
    }
    formatter_plugin_0 = module_0.FormatterPlugin(**dict_0)
    str_2 = formatter_plugin_0.format_headers(float_0)
    str_3 = "Similar to --form, but always sends a multipart/form-data request (i.e., even without files)."
    converter_plugin_0 = module_0.ConverterPlugin(str_3)
    transport_plugin_0 = module_0.TransportPlugin()
    transport_plugin_0.get_adapter()


def test_case_6():
    float_0 = 2687.63
    str_0 = "format_options"
    none_type_0 = None
    str_1 = "+<"
    dict_0 = {
        str_0: none_type_0,
        str_1: none_type_0,
        str_0: none_type_0,
        str_0: none_type_0,
    }
    formatter_plugin_0 = module_0.FormatterPlugin(**dict_0)
    str_2 = formatter_plugin_0.format_headers(float_0)
    str_3 = "Similar to --form, but always sends a multipart/form-data request (i.e., even without files)."
    str_4 = "Return processed `headers`\n\n        :param headers: The headers as text.\n\n        "
    str_5 = formatter_plugin_0.format_metadata(str_4)
    converter_plugin_0 = module_0.ConverterPlugin(str_3)
    module_0.FormatterPlugin()


def test_case_7():
    float_0 = 2687.63
    str_0 = "format_options"
    none_type_0 = None
    str_1 = "+<"
    dict_0 = {
        str_0: none_type_0,
        str_1: none_type_0,
        str_0: none_type_0,
        str_0: none_type_0,
    }
    formatter_plugin_0 = module_0.FormatterPlugin(**dict_0)
    str_2 = formatter_plugin_0.format_headers(float_0)
    str_3 = "Similar to --form, but always sends a multipart/form-data request (i.e., even without files)."
    converter_plugin_0 = module_0.ConverterPlugin(str_3)
    transport_plugin_0 = module_0.TransportPlugin()
    str_4 = "m"
    bytes_0 = b"\xfa\xff\xea/\xbfq;\x0e\xe9\x92~\xdbv"
    str_5 = formatter_plugin_0.format_body(str_4, bytes_0)
    transport_plugin_0.get_adapter()
