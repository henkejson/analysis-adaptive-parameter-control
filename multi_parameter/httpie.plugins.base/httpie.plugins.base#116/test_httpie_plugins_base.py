# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    transport_plugin_0 = module_0.TransportPlugin()


def test_case_1():
    str_0 = ";=%Ha}2GtMtp8@"
    auth_plugin_0 = module_0.AuthPlugin()
    auth_plugin_0.get_auth(auth_plugin_0, str_0)


def test_case_2():
    transport_plugin_0 = module_0.TransportPlugin()
    transport_plugin_0.get_adapter()


def test_case_3():
    bool_0 = True
    converter_plugin_0 = module_0.ConverterPlugin(bool_0)


def test_case_4():
    str_0 = "b"
    bytes_0 = b"_q\x7f\xa6\xec\xf4D,\x0b\x94T\xe08"
    converter_plugin_0 = module_0.ConverterPlugin(str_0)
    converter_plugin_0.convert(bytes_0)


def test_case_5():
    module_0.FormatterPlugin()
