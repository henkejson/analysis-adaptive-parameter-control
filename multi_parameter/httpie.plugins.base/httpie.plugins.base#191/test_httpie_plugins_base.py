# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0
import builtins as module_1


def test_case_0():
    transport_plugin_0 = module_0.TransportPlugin()


def test_case_1():
    str_0 = "hI4TzI#W3^d.P!:?nJ1"
    auth_plugin_0 = module_0.AuthPlugin()
    auth_plugin_0.get_auth(str_0)


def test_case_2():
    transport_plugin_0 = module_0.TransportPlugin()
    transport_plugin_0.get_adapter()


def test_case_3():
    none_type_0 = None
    transport_plugin_0 = module_0.TransportPlugin()
    var_0 = module_0.ConverterPlugin(none_type_0)
    var_0.get_auth()


def test_case_4():
    module_0.FormatterPlugin()


def test_case_5():
    str_0 = ""
    auth_plugin_0 = module_0.AuthPlugin(*str_0)
    var_0 = module_0.ConverterPlugin(str_0)
    var_1 = module_1.object()
    var_0.convert(var_0)
