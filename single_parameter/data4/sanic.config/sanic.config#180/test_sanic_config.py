# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.config as module_0
import inspect as module_1


def test_case_0():
    config_0 = module_0.Config()


def test_case_1():
    bool_0 = False
    none_type_0 = None
    config_0 = module_0.Config(none_type_0, keep_alive=bool_0, converters=none_type_0)


def test_case_2():
    int_0 = 2300
    var_0 = module_1.isdatadescriptor(int_0)
    config_0 = module_0.Config(env_prefix=var_0, converters=var_0)


def test_case_3():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)
    config_1 = module_0.Config(config_0, keep_alive=config_0)
    config_1.__getattr__(config_1)


def test_case_4():
    str_0 = "Set the worker to serving.\n\n        Args:\n            serving (bool): Whether the worker is serving.\n        "
    none_type_0 = None
    config_0 = module_0.Config(none_type_0, str_0)
    none_type_1 = config_0.update()
    config_0.__getattr__(none_type_1)


def test_case_5():
    config_0 = module_0.Config()
    none_type_0 = config_0.update()
    config_0.update_config(none_type_0)


def test_case_6():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)


def test_case_7():
    bool_0 = True
    module_0.Config(env_prefix=bool_0)


def test_case_8():
    none_type_0 = None
    config_0 = module_0.Config(env_prefix=none_type_0)
    str_0 = "Q\n|2~ "
    none_type_1 = config_0.__setitem__(str_0, str_0)
    config_0.update_config(none_type_0)


def test_case_9():
    none_type_0 = None
    config_0 = module_0.Config(none_type_0)
    none_type_1 = config_0.register_type(config_0)


def test_case_10():
    str_0 = "Set the worker to serving.\n\n        Args:\n            serving (bool): Whether the worker is serving.\n        "
    none_type_0 = None
    config_0 = module_0.Config(env_prefix=str_0, converters=none_type_0)
    module_0.Config(converters=str_0)


def test_case_11():
    config_0 = module_0.Config()
    config_0.__getattr__(config_0)


def test_case_12():
    none_type_0 = None
    config_0 = module_0.Config(keep_alive=none_type_0)
    config_1 = module_0.Config()
    config_2 = module_0.Config(converters=config_1)


def test_case_13():
    str_0 = "Set the worker to serving.\n\n        Args:\n            serving (bool): Whether the worker is serving.\n        "
    none_type_0 = None
    config_0 = module_0.Config(keep_alive=none_type_0)
    config_0.update_config(str_0)


def test_case_14():
    bool_0 = False
    none_type_0 = None
    config_0 = module_0.Config(none_type_0, keep_alive=bool_0, converters=none_type_0)
    str_0 = "LOCAL_CERT_CREATOR"
    config_0.__setattr__(str_0, none_type_0)
