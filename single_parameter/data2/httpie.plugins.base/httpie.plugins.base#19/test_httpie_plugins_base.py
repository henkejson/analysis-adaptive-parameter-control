# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    module_0.FormatterPlugin()


def test_case_1():
    auth_plugin_0 = module_0.AuthPlugin()
    none_type_0 = None
    auth_plugin_0.get_auth(password=none_type_0)


def test_case_2():
    transport_plugin_0 = module_0.TransportPlugin()
    transport_plugin_0.get_adapter()


def test_case_3():
    str_0 = "x6z/3I~1dr"
    str_1 = 'g^""R/'
    converter_plugin_0 = module_0.ConverterPlugin(str_1)
    module_0.TransportPlugin(*str_0)


def test_case_4():
    str_0 = ",k&\\k<H-\x0c"
    str_1 = "Positional arguments"
    converter_plugin_0 = module_0.ConverterPlugin(str_1)
    converter_plugin_0.convert(str_0)
