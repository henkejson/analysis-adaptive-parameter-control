# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    transport_plugin_0 = module_0.TransportPlugin()


def test_case_1():
    auth_plugin_0 = module_0.AuthPlugin()
    auth_plugin_0.get_auth()


def test_case_2():
    transport_plugin_0 = module_0.TransportPlugin()
    transport_plugin_0.get_adapter()


def test_case_3():
    str_0 = "O,Q"
    str_1 = "a22\rXB4zhy\r/3#{p"
    tuple_0 = (str_0, str_1)
    converter_plugin_0 = module_0.ConverterPlugin(tuple_0)


def test_case_4():
    module_0.FormatterPlugin()


def test_case_5():
    set_0 = set()
    str_0 = "!Ss{yYk;\\"
    converter_plugin_0 = module_0.ConverterPlugin(str_0)
    converter_plugin_0.convert(set_0)
