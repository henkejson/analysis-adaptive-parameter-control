# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.config as module_0
import inspect as module_1
import sanic.constants as module_2


def test_case_0():
    config_0 = module_0.Config()


def test_case_1():
    float_0 = -2030.269
    set_0 = {float_0}
    config_0 = module_0.Config(converters=set_0)
    config_1 = module_0.Config()
    var_0 = config_1.load_environment_vars()
    config_1.update_config(var_0)


def test_case_2():
    config_0 = module_0.Config()
    none_type_0 = None
    var_0 = config_0.load_environment_vars()
    var_1 = config_0.update_config(config_0)
    config_1 = module_0.Config(config_0, keep_alive=config_0, converters=none_type_0)
    config_0.update_config(var_1)


def test_case_3():
    none_type_0 = None
    config_0 = module_0.Config(none_type_0, none_type_0, converters=none_type_0)
    var_0 = module_1.getmembers(none_type_0)
    var_0.__getattr__(var_0)


def test_case_4():
    config_0 = module_0.Config()
    config_0.__getattr__(config_0)


def test_case_5():
    str_0 = "#\x0cN#\tm.oIQ"
    module_0.Config(converters=str_0)


def test_case_6():
    config_0 = module_0.Config()
    none_type_0 = None
    local_cert_creator_0 = module_2.LocalCertCreator.AUTO
    none_type_1 = config_0.__setitem__(local_cert_creator_0, local_cert_creator_0)
    none_type_2 = None
    var_0 = config_0.update_config(config_0)
    config_1 = module_0.Config(
        none_type_2, keep_alive=none_type_2, converters=none_type_0
    )
    config_0.update_config(var_0)


def test_case_7():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)


def test_case_8():
    none_type_0 = None
    config_0 = module_0.Config()
    config_0.update_config(none_type_0)


def test_case_9():
    config_0 = module_0.Config()
    none_type_0 = None
    module_0.Config(env_prefix=config_0, converters=none_type_0)


def test_case_10():
    config_0 = module_0.Config()
    config_0.create_empty_request()


def test_case_11():
    config_0 = module_0.Config()
    none_type_0 = None
    config_1 = module_0.Config(config_0, keep_alive=none_type_0, converters=none_type_0)


def test_case_12():
    config_0 = module_0.Config()
    none_type_0 = None
    config_1 = module_0.Config()
    config_2 = module_0.Config(config_1, keep_alive=none_type_0, converters=none_type_0)
    bytes_0 = b"\xd1\xfbl\xebq\xd4\xaf\xd0\xc55p\x9eL\xc9\x08m\xe7\xd0\xb3\xf7"
    config_0.update_config(bytes_0)
