# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.config as module_0
import sanic.constants as module_1
import sanic.errorpages as module_2


def test_case_0():
    config_0 = module_0.Config()


def test_case_1():
    float_0 = -2119.53926
    none_type_0 = None
    config_0 = module_0.Config(converters=none_type_0)
    config_0.__getattr__(float_0)


def test_case_2():
    str_0 = "G"
    config_0 = module_0.Config(env_prefix=str_0)


def test_case_3():
    bool_0 = True
    module_0.Config(converters=bool_0)


def test_case_4():
    none_type_0 = None
    config_0 = module_0.Config(converters=none_type_0)
    config_0.update_config(none_type_0)


def test_case_5():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)


def test_case_6():
    str_0 = ";2~lm:3PS&eI!so"
    config_0 = module_0.Config(env_prefix=str_0, keep_alive=str_0)
    config_1 = module_0.Config()
    var_0 = config_1.update_config(config_1)


def test_case_7():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)
    config_1 = module_0.Config()
    config_2 = module_0.Config(converters=config_0)
    float_0 = -2119.53926
    module_0.Config(float_0)


def test_case_8():
    config_0 = module_0.Config()
    config_0.__getattr__(config_0)


def test_case_9():
    config_0 = module_0.Config()
    config_1 = module_0.Config()
    str_0 = "b6c#fs+}IS1=14&L'"
    module_0.Config(converters=str_0)


def test_case_10():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)
    config_1 = module_0.Config(converters=config_0)
    float_0 = -2119.53926
    none_type_0 = config_1.register_type(float_0)


def test_case_11():
    config_0 = module_0.Config()
    config_1 = module_0.Config()
    local_cert_creator_0 = module_1.LocalCertCreator.MKCERT
    config_0.update_config(local_cert_creator_0)


def test_case_12():
    bool_0 = True
    config_0 = module_0.Config()
    config_1 = module_0.Config()
    config_2 = module_0.Config(converters=config_0)
    none_type_0 = None
    config_3 = module_0.Config(none_type_0, none_type_0)
    var_0 = config_3.update_config(config_3)
    var_1 = var_0.__repr__()
    module_2.check_error_format(bool_0)


def test_case_13():
    config_0 = module_0.Config()
    str_0 = "LOCAL_CERT_CREATOR"
    config_0.__setitem__(str_0, str_0)


def test_case_14():
    str_0 = "_FALLBACK_ERROR_FORMAT"
    str_1 = "FORWARDED_FOR_HEADER"
    int_0 = -2246
    dict_0 = {str_0: str_0, str_1: str_0, str_1: int_0, str_1: str_1}
    module_0.Config(dict_0)


def test_case_15():
    bool_0 = False
    config_0 = module_0.Config()
    str_0 = "_"
    config_1 = module_0.Config(config_0, str_0, bool_0)
    var_0 = config_1.update_config(config_1)
    config_0.update_config(bool_0)


def test_case_16():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)
    str_0 = "G"
    module_0.Config(var_0, str_0, var_0, converters=str_0)
