# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.config as module_0
import sanic.constants as module_1
import inspect as module_2
import sanic.helpers as module_3


def test_case_0():
    config_0 = module_0.Config()


def test_case_1():
    config_0 = module_0.Config()
    var_0 = module_1.LocalCertCreator.TRUSTME
    none_type_0 = config_0.__setitem__(var_0, config_0)
    var_1 = config_0.update_config(config_0)
    config_1 = module_0.Config(converters=config_0)


def test_case_2():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)
    var_1 = var_0.__bool__()
    str_0 = "bdM;"
    module_0.Config(str_0, converters=var_0)


def test_case_3():
    config_0 = module_0.Config()
    config_0.__getattr__(config_0)


def test_case_4():
    str_0 = "MFN*4}"
    list_0 = [str_0, str_0]
    module_0.Config(converters=list_0)


def test_case_5():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)
    config_1 = module_0.Config(env_prefix=var_0)


def test_case_6():
    config_0 = module_0.Config()
    var_0 = module_2.getmembers(config_0)
    config_0.update_config(var_0)


def test_case_7():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)


def test_case_8():
    config_0 = module_0.Config()
    config_0.__setitem__(config_0, config_0)


def test_case_9():
    config_0 = module_0.Config()
    local_cert_creator_0 = module_1.LocalCertCreator.TRUSTME
    none_type_0 = config_0.__setitem__(local_cert_creator_0, config_0)
    var_0 = config_0.update_config(config_0)
    config_0.update_config(local_cert_creator_0)


def test_case_10():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)
    config_0.update_config(var_0)


def test_case_11():
    str_0 = "H"
    config_0 = module_0.Config(env_prefix=str_0)


def test_case_12():
    default_0 = module_3.Default()
    tuple_0 = (default_0,)
    none_type_0 = None
    config_0 = module_0.Config(keep_alive=tuple_0, converters=none_type_0)
    config_0.__setattr__(default_0, config_0)


def test_case_13():
    str_0 = "http.middleware.after"
    str_1 = "_FALLBACK_ERROR_FORMAT"
    dict_0 = {str_0: str_0, str_1: str_1, str_1: str_1}
    module_0.Config(dict_0)


def test_case_14():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)
    none_type_0 = config_0.register_type(var_0)
    str_0 = "H"
    config_0.load_environment_vars(str_0)


def test_case_15():
    config_0 = module_0.Config()
    str_0 = "LOCAL_CERT_CREATOR"
    config_0.__setattr__(str_0, str_0)


def test_case_16():
    config_0 = module_0.Config()
    none_type_0 = config_0.update()
    var_0 = config_0.update_config(config_0)
    str_0 = "_"
    var_1 = config_0.load_environment_vars(str_0)
    config_0.update(*var_0, **var_1)
