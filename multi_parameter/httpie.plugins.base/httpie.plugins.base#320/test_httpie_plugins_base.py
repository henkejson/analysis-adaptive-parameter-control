# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    module_0.FormatterPlugin()


def test_case_1():
    none_type_0 = None
    list_0 = []
    auth_plugin_0 = module_0.AuthPlugin(*list_0)
    auth_plugin_0.get_auth(password=none_type_0)


def test_case_2():
    transport_plugin_0 = module_0.TransportPlugin()
    transport_plugin_0.get_adapter()


def test_case_3():
    none_type_0 = None
    str_0 = "--raw"
    converter_plugin_0 = module_0.ConverterPlugin(none_type_0)
    str_1 = "replace"
    dict_0 = {str_0: none_type_0, str_1: none_type_0}
    module_0.FormatterPlugin(**dict_0)


def test_case_4():
    str_0 = "-sGH \rDm\t*V "
    str_1 = "]z_;SZDUi%~}1(;\x0brWCm"
    converter_plugin_0 = module_0.ConverterPlugin(str_1)
    converter_plugin_0.convert(str_0)
