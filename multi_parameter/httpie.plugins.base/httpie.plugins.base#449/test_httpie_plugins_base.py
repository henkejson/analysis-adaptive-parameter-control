# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    transport_plugin_0 = module_0.TransportPlugin()


def test_case_1():
    bytes_0 = b"Z@\xfd\xe3/\xb0H\xe8\xcd\xac\xc1%Q_(\xf19\xa0\x83"
    auth_plugin_0 = module_0.AuthPlugin()
    auth_plugin_0.get_auth(bytes_0)


def test_case_2():
    base_plugin_0 = module_0.BasePlugin()
    transport_plugin_0 = module_0.TransportPlugin()
    transport_plugin_0.get_adapter()


def test_case_3():
    str_0 = "#FCBFB"
    converter_plugin_0 = module_0.ConverterPlugin(str_0)


def test_case_4():
    str_0 = "#FCDBFC"
    converter_plugin_0 = module_0.ConverterPlugin(str_0)
    converter_plugin_0.convert(str_0)


def test_case_5():
    module_0.FormatterPlugin()
