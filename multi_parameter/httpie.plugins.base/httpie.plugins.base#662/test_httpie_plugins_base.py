# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    module_0.FormatterPlugin()


def test_case_1():
    auth_plugin_0 = module_0.AuthPlugin()
    auth_plugin_0.get_auth()


def test_case_2():
    transport_plugin_0 = module_0.TransportPlugin()
    transport_plugin_0.get_adapter()


def test_case_3():
    str_0 = "."
    tuple_0 = (str_0, str_0)
    converter_plugin_0 = module_0.ConverterPlugin(tuple_0)
    str_1 = "500"
    none_type_0 = None
    str_2 = ":<2H'Kj"
    str_3 = "jbKCKr`4=-a[,"
    dict_0 = {
        str_1: none_type_0,
        str_2: none_type_0,
        str_3: none_type_0,
        str_1: none_type_0,
    }
    module_0.TransportPlugin(**dict_0)


def test_case_4():
    bytes_0 = b"\xb6\xf7l1\xe3\xdc\xbe\xad_[\xc0d\xb4eR"
    converter_plugin_0 = module_0.ConverterPlugin(bytes_0)
    bytes_1 = b"\xb8\x04"
    converter_plugin_0.convert(bytes_1)
