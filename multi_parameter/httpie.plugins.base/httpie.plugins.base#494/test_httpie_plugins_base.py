# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    module_0.FormatterPlugin()


def test_case_1():
    auth_plugin_0 = module_0.AuthPlugin()
    list_0 = [auth_plugin_0, auth_plugin_0, auth_plugin_0, auth_plugin_0]
    auth_plugin_0.get_auth(password=list_0)


def test_case_2():
    list_0 = []
    dict_0 = {}
    transport_plugin_0 = module_0.TransportPlugin(*list_0, **dict_0)
    transport_plugin_0.get_adapter()


def test_case_3():
    none_type_0 = None
    str_0 = "Q"
    str_1 = "Content-"
    tuple_0 = (str_0, str_1)
    set_0 = {str_0, tuple_0, str_1}
    tuple_1 = (tuple_0, set_0)
    converter_plugin_0 = module_0.ConverterPlugin(tuple_1)
    converter_plugin_0.convert(none_type_0)
