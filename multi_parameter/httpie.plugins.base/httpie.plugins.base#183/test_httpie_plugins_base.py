# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    auth_plugin_0 = module_0.AuthPlugin()


def test_case_1():
    str_0 = "Q [DM"
    auth_plugin_0 = module_0.AuthPlugin()
    auth_plugin_0.get_auth(password=str_0)


def test_case_2():
    transport_plugin_0 = module_0.TransportPlugin()
    transport_plugin_0.get_adapter()


def test_case_3():
    dict_0 = {}
    transport_plugin_0 = module_0.TransportPlugin(**dict_0)
    str_0 = "P5Qqly,>"
    converter_plugin_0 = module_0.ConverterPlugin(str_0)
    bytes_0 = b"\x1d5\xde=\x1d\x17*#Db[\xf9g"
    module_0.FormatterPlugin(**bytes_0)


def test_case_4():
    str_0 = "`\\EKD=A<Nb'9*t<-u<"
    converter_plugin_0 = module_0.ConverterPlugin(str_0)
    converter_plugin_0.convert(str_0)


def test_case_5():
    module_0.FormatterPlugin()
