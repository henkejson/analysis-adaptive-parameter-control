# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.config as module_0
import inspect as module_1
import abc as module_2


def test_case_0():
    config_0 = module_0.Config()


def test_case_1():
    bool_0 = False
    config_0 = module_0.Config(keep_alive=bool_0)


def test_case_2():
    none_type_0 = None
    config_0 = module_0.Config(env_prefix=none_type_0)
    config_1 = module_0.Config(config_0, keep_alive=config_0, converters=config_0)


def test_case_3():
    str_0 = "\n        Read and stream the body in chunks from an incoming ASGI message.\n        "
    module_0.Config(env_prefix=str_0, converters=str_0)


def test_case_4():
    config_0 = module_0.Config()
    var_0 = module_1.getmembers(config_0)
    config_0.update_config(var_0)


def test_case_5():
    bool_0 = False
    config_0 = module_0.Config()
    config_0.update_config(bool_0)


def test_case_6():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)


def test_case_7():
    str_0 = "\n        Read and stream the body in chunks from an incoming ASGI message.\n        "
    list_0 = []
    var_0 = module_1.isdatadescriptor(list_0)
    config_0 = module_0.Config(env_prefix=str_0, converters=var_0)


def test_case_8():
    config_0 = module_0.Config()
    str_0 = "L"
    none_type_0 = None
    config_1 = module_0.Config(env_prefix=str_0, converters=none_type_0)
    module_2.ABCMeta()


def test_case_9():
    str_0 = "\n        Read and stream the ody in chunks from an incomin ASGI message.\n        "
    list_0 = []
    config_0 = module_0.Config(keep_alive=list_0)
    config_0.update_config(str_0)


def test_case_10():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)
    config_0.__getattr__(config_0)


def test_case_11():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)
    str_0 = "LOCAL_CERT_CREATOR"
    config_0.__setitem__(str_0, config_0)


def test_case_12():
    str_0 = "L"
    module_0.Config(env_prefix=str_0, converters=str_0)
