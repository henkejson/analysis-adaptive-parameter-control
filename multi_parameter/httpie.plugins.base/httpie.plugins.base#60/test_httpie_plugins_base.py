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
    base_plugin_0 = module_0.BasePlugin(*list_0)
    transport_plugin_0 = module_0.TransportPlugin()
    transport_plugin_0.get_adapter()


def test_case_3():
    auth_plugin_0 = module_0.AuthPlugin()
    var_0 = module_0.ConverterPlugin(auth_plugin_0)
    var_0.get_adapter()


def test_case_4():
    str_0 = "hvmV8-W'N6\n<~)dk*"
    str_1 = "B@H-LcL9av"
    tuple_0 = (str_0, str_1)
    str_2 = ".N;;Lx3)xG."
    converter_plugin_0 = module_0.ConverterPlugin(str_2)
    converter_plugin_0.convert(tuple_0)
