# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.config as module_0
import inspect as module_1
import sanic.http.stream as module_2


def test_case_0():
    config_0 = module_0.Config()


def test_case_1():
    bool_0 = False
    config_0 = module_0.Config(bool_0, keep_alive=bool_0)
    str_0 = '*oqLm"`>'
    none_type_0 = None
    none_type_1 = config_0.__setattr__(str_0, none_type_0)
    list_0 = [config_0]
    none_type_2 = config_0.update(*list_0)


def test_case_2():
    str_0 = ">a_OY0zU"
    bool_0 = False
    dict_0 = {str_0: bool_0}
    config_0 = module_0.Config(dict_0, str_0)
    var_0 = config_0.load_environment_vars()
    config_1 = module_0.Config()
    list_0 = [config_1, config_1, config_1]
    none_type_0 = None
    none_type_1 = config_1.register_type(none_type_0)
    none_type_2 = config_1.update(*list_0)
    none_type_3 = config_1.register_type(config_1)
    var_1 = config_1.update_config(config_1)
    config_1.update_config(var_1)


def test_case_3():
    float_0 = 3228.3
    str_0 = "U"
    module_0.Config(float_0, str_0)


def test_case_4():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)
    config_0.__getattr__(var_0)


def test_case_5():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)
    config_0.__getattr__(config_0)


def test_case_6():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)


def test_case_7():
    bytes_0 = b"Reset Content"
    bool_0 = False
    str_0 = "*_oynP"
    config_0 = module_0.Config(bool_0, str_0)
    config_0.update_config(bytes_0)


def test_case_8():
    config_0 = module_0.Config()
    none_type_0 = config_0.register_type(config_0)
    var_0 = config_0.update_config(config_0)


def test_case_9():
    none_type_0 = None
    bytes_0 = b"\x08\x83%/E\xf2_nA\xc9\tB\x0b"
    config_0 = module_0.Config(env_prefix=none_type_0)
    var_0 = module_1.getmembers(bytes_0)
    module_0.DescriptorMeta(none_type_0)


def test_case_10():
    config_0 = module_0.Config()
    none_type_0 = None
    config_0.update_config(none_type_0)


def test_case_11():
    config_0 = module_0.Config()
    list_0 = [config_0, config_0, config_0]
    str_0 = '\n@o$Ebm;SX1]"re<ERRV'
    none_type_0 = None
    config_1 = module_0.Config(config_0, str_0, none_type_0)
    none_type_1 = None
    none_type_2 = config_0.register_type(none_type_1)
    none_type_3 = config_0.update(*list_0)
    none_type_4 = config_0.register_type(config_0)
    str_1 = "LOCAL_CERT_CREATOR"
    config_0.__setitem__(str_1, none_type_4)


def test_case_12():
    config_0 = module_0.Config()
    none_type_0 = config_0.register_type(config_0)
    config_0.register_type(config_0)


def test_case_13():
    config_0 = module_0.Config()
    none_type_0 = config_0.register_type(config_0)
    config_1 = module_0.Config(keep_alive=none_type_0, converters=config_0)
    var_0 = config_0.update_config(config_0)
    module_2.Stream(**config_0)


def test_case_14():
    str_0 = "_"
    none_type_0 = None
    config_0 = module_0.Config(env_prefix=str_0, converters=none_type_0)
    none_type_1 = None
    var_0 = module_1.getmembers(none_type_1)
    var_0.register_type(var_0)


def test_case_15():
    config_0 = module_0.Config()
    str_0 = "FALLBACK_ERROR_FORMAT"
    config_0.__setitem__(str_0, str_0)
