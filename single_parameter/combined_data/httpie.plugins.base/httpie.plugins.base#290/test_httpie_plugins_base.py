# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    base_plugin_0 = module_0.BasePlugin()


def test_case_1():
    str_0 = "(w_w9^1M"
    auth_plugin_0 = module_0.AuthPlugin()
    auth_plugin_0.get_auth(str_0)


def test_case_2():
    transport_plugin_0 = module_0.TransportPlugin()
    transport_plugin_0.get_adapter()


def test_case_3():
    bool_0 = True
    set_0 = {bool_0, bool_0, bool_0}
    converter_plugin_0 = module_0.ConverterPlugin(set_0)


def test_case_4():
    str_0 = "-I"
    converter_plugin_0 = module_0.ConverterPlugin(str_0)
    bytes_0 = b"-\x0f\x06\x13>\xab\xaf3\xb5["
    converter_plugin_0.convert(bytes_0)


def test_case_5():
    module_0.FormatterPlugin()
