# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    base_plugin_0 = module_0.BasePlugin()


def test_case_1():
    auth_plugin_0 = module_0.AuthPlugin()
    auth_plugin_0.get_auth()


def test_case_2():
    transport_plugin_0 = module_0.TransportPlugin()
    transport_plugin_0.get_adapter()


def test_case_3():
    str_0 = "("
    converter_plugin_0 = module_0.ConverterPlugin(str_0)
    none_type_0 = None
    module_0.AuthPlugin(*converter_plugin_0, **none_type_0)


def test_case_4():
    str_0 = "4@#h6zBNE^#ai4,\\|%"
    converter_plugin_0 = module_0.ConverterPlugin(str_0)
    converter_plugin_0.convert(str_0)


def test_case_5():
    module_0.FormatterPlugin()
