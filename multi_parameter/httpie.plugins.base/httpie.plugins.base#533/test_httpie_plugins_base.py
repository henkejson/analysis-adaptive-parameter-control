# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    transport_plugin_0 = module_0.TransportPlugin()


def test_case_1():
    str_0 = "index"
    auth_plugin_0 = module_0.AuthPlugin()
    auth_plugin_0.get_auth(password=str_0)


def test_case_2():
    transport_plugin_0 = module_0.TransportPlugin()
    transport_plugin_0.get_adapter()


def test_case_3():
    str_0 = "mDM$6>=/n;M];{3fXf"
    bytes_0 = b"c>3k\xc1\xfa\xc9"
    converter_plugin_0 = module_0.ConverterPlugin(str_0)
    converter_plugin_0.convert(bytes_0)


def test_case_4():
    module_0.FormatterPlugin()
