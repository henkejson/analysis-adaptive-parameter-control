# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    transport_plugin_0 = module_0.TransportPlugin()


def test_case_1():
    auth_plugin_0 = module_0.AuthPlugin()
    auth_plugin_0.get_auth(password=auth_plugin_0)


def test_case_2():
    auth_plugin_0 = module_0.AuthPlugin()
    transport_plugin_0 = module_0.TransportPlugin()
    transport_plugin_0.get_adapter()


def test_case_3():
    dict_0 = {}
    auth_plugin_0 = module_0.AuthPlugin(**dict_0)
    converter_plugin_0 = module_0.ConverterPlugin(dict_0)
    list_0 = [auth_plugin_0, auth_plugin_0, auth_plugin_0]
    module_0.TransportPlugin(*list_0)


def test_case_4():
    str_0 = "ssl3"
    bool_0 = True
    converter_plugin_0 = module_0.ConverterPlugin(bool_0)
    converter_plugin_0.convert(str_0)


def test_case_5():
    module_0.FormatterPlugin()


def test_case_6():
    auth_plugin_0 = module_0.AuthPlugin()
    transport_plugin_0 = module_0.TransportPlugin()
    none_type_0 = None
    str_0 = ':9[Kb}^/_6ijMTO")'
    str_1 = "line"
    str_2 = "format_options"
    dict_0 = {str_1: str_1, str_2: str_2, str_1: str_2}
    formatter_plugin_0 = module_0.FormatterPlugin(**dict_0)
    str_3 = formatter_plugin_0.format_body(none_type_0, str_0)
    module_0.TransportPlugin(**dict_0)


def test_case_7():
    auth_plugin_0 = module_0.AuthPlugin()
    transport_plugin_0 = module_0.TransportPlugin()
    none_type_0 = None
    str_0 = "w<q;\r3`9Z`;=|"
    str_1 = "format_options"
    dict_0 = {str_0: str_0, str_1: str_1, str_0: str_1}
    formatter_plugin_0 = module_0.FormatterPlugin(**dict_0)
    str_2 = "nflQ7k]-X"
    str_3 = formatter_plugin_0.format_body(none_type_0, str_2)
    str_4 = formatter_plugin_0.format_metadata(str_2)
    auth_plugin_0.get_auth(password=none_type_0)


def test_case_8():
    auth_plugin_0 = module_0.AuthPlugin()
    transport_plugin_0 = module_0.TransportPlugin()
    none_type_0 = None
    str_0 = ":[Kbf^/_6ijMTOu)"
    str_1 = "w<q;\r3`9Z`;=|"
    str_2 = "format_options"
    dict_0 = {str_1: str_1, str_2: str_2, str_2: str_2}
    formatter_plugin_0 = module_0.FormatterPlugin(**dict_0)
    str_3 = formatter_plugin_0.format_body(none_type_0, str_1)
    str_4 = formatter_plugin_0.format_metadata(str_3)
    str_5 = formatter_plugin_0.format_body(str_0, str_0)
    str_6 = formatter_plugin_0.format_body(none_type_0, str_4)
    auth_plugin_1 = module_0.AuthPlugin()
    str_7 = formatter_plugin_0.format_body(str_3, str_0)
    str_8 = formatter_plugin_0.format_headers(str_6)
    auth_plugin_1.get_auth(str_8)
