# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    module_0.FormatterPlugin()


def test_case_1():
    str_0 = "HTTPIE_CONFIG_DIR"
    auth_plugin_0 = module_0.AuthPlugin()
    auth_plugin_0.get_auth(password=str_0)


def test_case_2():
    transport_plugin_0 = module_0.TransportPlugin()
    transport_plugin_0.get_adapter()


def test_case_3():
    none_type_0 = None
    converter_plugin_0 = module_0.ConverterPlugin(none_type_0)


def test_case_4():
    str_0 = "akYS"
    tuple_0 = (str_0, str_0)
    str_1 = "passphrase for "
    dict_0 = {str_1: str_1}
    converter_plugin_0 = module_0.ConverterPlugin(dict_0)
    converter_plugin_0.convert(tuple_0)
