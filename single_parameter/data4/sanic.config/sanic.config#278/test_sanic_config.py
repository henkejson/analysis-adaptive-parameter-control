# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.config as module_0
import sanic.http.stream as module_1
import sanic.constants as module_2


def test_case_0():
    config_0 = module_0.Config()


def test_case_1():
    bool_0 = True
    config_0 = module_0.Config(keep_alive=bool_0)


def test_case_2():
    none_type_0 = None
    config_0 = module_0.Config(env_prefix=none_type_0)


def test_case_3():
    none_type_0 = None
    config_0 = module_0.Config(keep_alive=none_type_0)
    str_0 = ".*"
    module_0.Config(str_0)


def test_case_4():
    none_type_0 = None
    config_0 = module_0.Config(keep_alive=none_type_0)
    str_0 = ".*"
    none_type_1 = config_0.__setattr__(str_0, none_type_0)
    var_0 = config_0.load_environment_vars(str_0)
    config_0.__getattr__(none_type_0)


def test_case_5():
    none_type_0 = None
    config_0 = module_0.Config(env_prefix=none_type_0)
    config_0.__getattr__(config_0)


def test_case_6():
    stream_0 = module_1.Stream()
    none_type_0 = None
    config_0 = module_0.Config(keep_alive=none_type_0)
    var_0 = config_0.update_config(config_0)


def test_case_7():
    str_0 = "~\ti~]t6Eo$"
    none_type_0 = None
    config_0 = module_0.Config(env_prefix=none_type_0, converters=none_type_0)
    config_0.update_config(str_0)


def test_case_8():
    none_type_0 = None
    config_0 = module_0.Config(keep_alive=none_type_0)
    config_0.update_config(none_type_0)


def test_case_9():
    none_type_0 = None
    config_0 = module_0.Config(keep_alive=none_type_0)
    none_type_1 = config_0.register_type(config_0)


def test_case_10():
    int_0 = -842
    module_0.Config(env_prefix=int_0, keep_alive=int_0)


def test_case_11():
    none_type_0 = None
    str_0 = "P"
    config_0 = module_0.Config(none_type_0, str_0)


def test_case_12():
    str_0 = "XfEV(nK"
    config_0 = module_0.Config(converters=str_0)
    module_0.DescriptorMeta(config_0)


def test_case_13():
    none_type_0 = None
    config_0 = module_0.Config(keep_alive=none_type_0)
    var_0 = config_0.update_config(config_0)
    var_1 = config_0.update_config(config_0)
    var_2 = config_0.load_environment_vars()
    config_1 = module_0.Config(converters=config_0)
    local_cert_creator_0 = module_2.LocalCertCreator.MKCERT
    none_type_1 = config_0.register_type(config_0)
    none_type_2 = config_0.__setattr__(local_cert_creator_0, var_2)
    config_0.register_type(config_0)


def test_case_14():
    str_0 = "_FALLBACK_ERROR_FORMAT"
    str_1 = "}!o^:P"
    dict_0 = {str_0: str_0, str_1: str_0}
    none_type_0 = None
    module_0.Config(dict_0, none_type_0, none_type_0)
