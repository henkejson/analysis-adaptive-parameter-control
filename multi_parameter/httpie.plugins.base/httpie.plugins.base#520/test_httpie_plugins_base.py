# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    base_plugin_0 = module_0.BasePlugin()


def test_case_1():
    base_plugin_0 = module_0.BasePlugin()
    none_type_0 = None
    dict_0 = {}
    auth_plugin_0 = module_0.AuthPlugin(**dict_0)
    auth_plugin_0.get_auth(password=none_type_0)


def test_case_2():
    transport_plugin_0 = module_0.TransportPlugin()
    transport_plugin_0.get_adapter()


def test_case_3():
    str_0 = "Z6gy+mF|5/"
    converter_plugin_0 = module_0.ConverterPlugin(str_0)
    str_1 = ""
    str_2 = "FGaq#"
    none_type_0 = None
    str_3 = "Upgraded "
    str_4 = '"O64'
    dict_0 = {
        str_2: none_type_0,
        str_3: none_type_0,
        str_1: none_type_0,
        str_4: none_type_0,
    }
    module_0.FormatterPlugin(**dict_0)


def test_case_4():
    int_0 = 675
    float_0 = 377.0
    set_0 = {int_0, int_0, float_0, int_0}
    str_0 = 'W)" \x0b#h)Jhh42)J+'
    converter_plugin_0 = module_0.ConverterPlugin(str_0)
    converter_plugin_1 = module_0.ConverterPlugin(set_0)
    bytes_0 = b"\n\n"
    converter_plugin_1.convert(bytes_0)


def test_case_5():
    module_0.FormatterPlugin()
