# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    transport_plugin_0 = module_0.TransportPlugin()


def test_case_1():
    auth_plugin_0 = module_0.AuthPlugin()
    converter_plugin_0 = module_0.ConverterPlugin(auth_plugin_0)
    auth_plugin_0.get_auth()


def test_case_2():
    transport_plugin_0 = module_0.TransportPlugin()
    transport_plugin_0.get_adapter()


def test_case_3():
    bytes_0 = b"_\x80\xaabD\xca\x8d<\x87s\x83\x04\xc9"
    str_0 = "(7\x0cW3Z}hW~="
    converter_plugin_0 = module_0.ConverterPlugin(str_0)
    converter_plugin_0.convert(bytes_0)


def test_case_4():
    module_0.FormatterPlugin()
