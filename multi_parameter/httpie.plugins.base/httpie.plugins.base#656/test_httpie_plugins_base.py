# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import httpie.plugins.base as module_0


def test_case_0():
    module_0.FormatterPlugin()


def test_case_1():
    auth_plugin_0 = module_0.AuthPlugin()
    str_0 = "HTTPie/"
    auth_plugin_1 = module_0.AuthPlugin()
    auth_plugin_1.get_auth(password=str_0)


def test_case_2():
    transport_plugin_0 = module_0.TransportPlugin()
    transport_plugin_0.get_adapter()


def test_case_3():
    str_0 = "4{Hf}g0cC\x0bY#gl7w`E"
    converter_plugin_0 = module_0.ConverterPlugin(str_0)


def test_case_4():
    auth_plugin_0 = module_0.AuthPlugin()
    float_0 = 5068.503787
    converter_plugin_0 = module_0.ConverterPlugin(float_0)
    bytes_0 = b"\xb4\x17\xfeh\x9a\xa00\xce\xea|\x95\xaby>p5K\xab\x98\xcb"
    converter_plugin_0.convert(bytes_0)
