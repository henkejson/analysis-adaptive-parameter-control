# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    transport_plugin_0 = module_0.TransportPlugin()


def test_case_1():
    auth_plugin_0 = module_0.AuthPlugin()
    auth_plugin_0.get_auth(auth_plugin_0)


def test_case_2():
    transport_plugin_0 = module_0.TransportPlugin()
    transport_plugin_0.get_adapter()


def test_case_3():
    str_0 = "X81og7tfk]PB"
    none_type_0 = None
    converter_plugin_0 = module_0.ConverterPlugin(str_0)
    converter_plugin_0.convert(none_type_0)


def test_case_4():
    module_0.FormatterPlugin()


def test_case_5():
    str_0 = "module"
    str_1 = "format_options"
    dict_0 = {str_1: str_1}
    formatter_plugin_0 = module_0.FormatterPlugin(**dict_0)
    str_2 = formatter_plugin_0.format_headers(str_0)
    str_3 = formatter_plugin_0.format_headers(dict_0)
    str_4 = "\n    start: root_path path*\n    root_path: (literal | index_path | append_path)\n    literal: TEXT | NUMBER\n\n    path:\n        key_path\n        | index_path\n        | append_path\n    key_path: LEFT_BRACKET TEXT RIGHT_BRACKET\n    index_path: LEFT_BRACKET NUMBER RIGHT_BRACKET\n    append_path: LEFT_BRACKET RIGHT_BRACKET\n\n    "
    bool_0 = True
    list_0 = [bool_0, str_4]
    str_5 = "(%iZ#`4M9T5- <0}"
    dict_1 = {str_4: list_0, str_4: bool_0, str_5: list_0}
    module_0.FormatterPlugin(**dict_1)


def test_case_6():
    str_0 = "module"
    str_1 = "format_options"
    dict_0 = {str_1: str_1}
    formatter_plugin_0 = module_0.FormatterPlugin(**dict_0)
    str_2 = formatter_plugin_0.format_headers(str_0)
    str_3 = formatter_plugin_0.format_headers(dict_0)
    str_4 = "\n    start: root_path path*\n    root_path: (literal | index_path | append_path)\n    literal: TEXT | NUMBER\n\n    path:\n        key_path\n        | index_path\n        | append_path\n    key_path: LEFT_BRACKET TEXT RIGHT_BRACKET\n    index_path: LEFT_BRACKET NUMBER RIGHT_BRACKET\n    append_path: LEFT_BRACKET RIGHT_BRACKET\n\n    "
    bool_0 = True
    list_0 = [bool_0, str_4]
    str_5 = "(%iZ#`4M9T5- <0}"
    none_type_0 = None
    str_6 = formatter_plugin_0.format_body(list_0, none_type_0)
    dict_1 = {str_4: list_0, str_4: bool_0, str_5: list_0}
    module_0.FormatterPlugin(**dict_1)


def test_case_7():
    str_0 = "module"
    str_1 = "format_options"
    dict_0 = {str_1: str_1}
    formatter_plugin_0 = module_0.FormatterPlugin(**dict_0)
    str_2 = formatter_plugin_0.format_headers(str_0)
    str_3 = formatter_plugin_0.format_metadata(str_1)
    str_4 = formatter_plugin_0.format_headers(dict_0)
    str_5 = "\n    start: root_path path*\n    root_path: (literal | index_path | append_path)\n    literal: TEXT | NUMBER\n\n    path:\n        key_path\n        | index_path\n        | append_path\n    key_path: LEFT_BRACKET TEXT RIGHT_BRACKET\n    index_path: LEFT_BRACKET NUMBER RIGHT_BRACKET\n    append_path: LEFT_BRACKET RIGHT_BRACKET\n\n    "
    bool_0 = True
    list_0 = [bool_0, str_5]
    str_6 = "(%iZ#`4M9T5- <0}"
    dict_1 = {str_5: list_0, str_5: bool_0, str_6: list_0}
    module_0.FormatterPlugin(**dict_1)
