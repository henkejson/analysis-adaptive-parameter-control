# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.config as module_0
import sanic.errorpages as module_1
import inspect as module_2


def test_case_0():
    config_0 = module_0.Config()


def test_case_1():
    none_type_0 = None
    config_0 = module_0.Config(env_prefix=none_type_0)


def test_case_2():
    bool_0 = True
    module_0.Config(bool_0, bool_0)


def test_case_3():
    config_0 = module_0.Config()
    none_type_0 = None
    config_0.__getattr__(none_type_0)


def test_case_4():
    none_type_0 = None
    config_0 = module_0.Config(env_prefix=none_type_0)
    config_0.update_config(none_type_0)


def test_case_5():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)


def test_case_6():
    none_type_0 = None
    config_0 = module_0.Config(env_prefix=none_type_0)
    none_type_1 = config_0.register_type(none_type_0)
    config_0.update_config(none_type_0)


def test_case_7():
    str_0 = "S"
    config_0 = module_0.Config(env_prefix=str_0)
    config_1 = module_0.Config(keep_alive=config_0)
    bool_0 = True
    module_1.check_error_format(bool_0)


def test_case_8():
    config_0 = module_0.Config()
    config_0.__getattr__(config_0)


def test_case_9():
    config_0 = module_0.Config()
    none_type_0 = config_0.register_type(config_0)
    var_0 = module_2.isdatadescriptor(config_0)
    config_0.register_type(config_0)


def test_case_10():
    str_0 = "LOCAL_TLS_CERT"
    dict_0 = {str_0: str_0}
    config_0 = module_0.Config(dict_0, converters=dict_0)
    config_1 = module_0.Config()
    var_0 = config_1.update_config(config_1)
    none_type_0 = config_1.register_type(config_1)
    none_type_1 = config_1.update(**config_1)
    config_1.__getattr__(var_0)


def test_case_11():
    str_0 = "S"
    config_0 = module_0.Config(env_prefix=str_0)
    config_1 = module_0.Config()
    var_0 = config_1.update_config(config_0)
    config_1.update_config(str_0)


def test_case_12():
    str_0 = "_"
    config_0 = module_0.Config(env_prefix=str_0)
    config_1 = module_0.Config()
    none_type_0 = config_1.register_type(config_1)
    float_0 = 1804.971592896656
    var_0 = config_1.update_config(config_1)
    none_type_0.update(**float_0)
