# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    auth_plugin_0 = module_0.AuthPlugin()


def test_case_1():
    auth_plugin_0 = module_0.AuthPlugin()
    auth_plugin_0.get_auth()


def test_case_2():
    transport_plugin_0 = module_0.TransportPlugin()
    transport_plugin_0.get_adapter()


def test_case_3():
    str_0 = "M,<(JW\\1"
    converter_plugin_0 = module_0.ConverterPlugin(str_0)
    transport_plugin_0 = module_0.TransportPlugin()


def test_case_4():
    str_0 = "WO40)-[g@X$:.s"
    bytes_0 = b"\xe0\xd7\xd1\xc3\x05E_\x08Y\x8f\x81|"
    converter_plugin_0 = module_0.ConverterPlugin(str_0)
    converter_plugin_0.convert(bytes_0)


def test_case_5():
    module_0.FormatterPlugin()
