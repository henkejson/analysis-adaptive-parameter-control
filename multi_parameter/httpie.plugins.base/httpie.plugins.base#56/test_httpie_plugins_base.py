# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    module_0.FormatterPlugin()


def test_case_1():
    auth_plugin_0 = module_0.AuthPlugin()
    auth_plugin_0.get_auth()


def test_case_2():
    list_0 = []
    transport_plugin_0 = module_0.TransportPlugin(*list_0)
    transport_plugin_0.get_adapter()


def test_case_3():
    str_0 = " l{eT!7r?\t="
    converter_plugin_0 = module_0.ConverterPlugin(str_0)
    auth_plugin_0 = module_0.AuthPlugin()
    auth_plugin_0.get_auth(str_0, str_0)


def test_case_4():
    int_0 = -3036
    bytes_0 = b"\x01\xd7\x97ymB\x1c"
    converter_plugin_0 = module_0.ConverterPlugin(int_0)
    converter_plugin_0.convert(bytes_0)


def test_case_5():
    str_0 = "format_options"
    str_1 = "804pk:KAW\x0c_w7O#Y!-"
    bool_0 = False
    dict_0 = {str_0: str_0}
    formatter_plugin_0 = module_0.FormatterPlugin(**dict_0)
    str_2 = formatter_plugin_0.format_metadata(str_0)
    str_3 = "version_info_file"
    bool_1 = False
    dict_1 = {str_0: str_0, str_1: bool_0, str_0: str_0, str_3: bool_1}
    module_0.TransportPlugin(**dict_1)


def test_case_6():
    str_0 = "format_options"
    str_1 = "804pk:KAW\x0c_w7O#Y!-"
    bool_0 = False
    dict_0 = {str_0: str_0}
    formatter_plugin_0 = module_0.FormatterPlugin(**dict_0)
    str_2 = formatter_plugin_0.format_metadata(str_0)
    str_3 = "version_info_file"
    str_4 = "Hi!Rkb;4x RP"
    str_5 = formatter_plugin_0.format_body(str_2, str_4)
    bool_1 = False
    dict_1 = {str_0: str_0, str_1: bool_0, str_0: str_0, str_3: bool_1}
    module_0.TransportPlugin(**dict_1)


def test_case_7():
    str_0 = "format_options"
    str_1 = "804pk:KAW\x0c_w7O#Y!-"
    bool_0 = False
    dict_0 = {str_0: str_0}
    formatter_plugin_0 = module_0.FormatterPlugin(**dict_0)
    str_2 = formatter_plugin_0.format_headers(bool_0)
    str_3 = formatter_plugin_0.format_metadata(str_0)
    str_4 = "version_info_file"
    bool_1 = False
    dict_1 = {str_0: str_0, str_1: bool_0, str_0: str_0, str_4: bool_1}
    module_0.TransportPlugin(**dict_1)
