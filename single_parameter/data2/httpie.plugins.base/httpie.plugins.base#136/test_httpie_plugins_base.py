# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    module_0.FormatterPlugin()


def test_case_1():
    str_0 = "__hack__"
    auth_plugin_0 = module_0.AuthPlugin()
    auth_plugin_0.get_auth(password=str_0)


def test_case_2():
    transport_plugin_0 = module_0.TransportPlugin()
    transport_plugin_0.get_adapter()


def test_case_3():
    bool_0 = True
    converter_plugin_0 = module_0.ConverterPlugin(bool_0)
    list_0 = []
    str_0 = "SHS aM$#.o"
    str_1 = "27U7"
    dict_0 = {str_0: converter_plugin_0, str_1: list_0, str_1: str_1, str_0: bool_0}
    module_0.TransportPlugin(*list_0, **dict_0)


def test_case_4():
    str_0 = "\nt@>mIE=1uP2"
    auth_plugin_0 = module_0.AuthPlugin()
    var_0 = module_0.ConverterPlugin(str_0)
    var_0.convert(var_0)
