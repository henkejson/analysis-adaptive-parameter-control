# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.config as module_0
import inspect as module_1


def test_case_0():
    config_0 = module_0.Config()


def test_case_1():
    none_type_0 = None
    config_0 = module_0.Config(env_prefix=none_type_0)
    config_1 = module_0.Config(keep_alive=none_type_0, converters=config_0)
    config_0.update_config(none_type_0)


def test_case_2():
    bool_0 = False
    config_0 = module_0.Config(keep_alive=bool_0)


def test_case_3():
    none_type_0 = None
    config_0 = module_0.Config(env_prefix=none_type_0, keep_alive=none_type_0)


def test_case_4():
    config_0 = module_0.Config()
    none_type_0 = config_0.update(**config_0)
    str_0 = "Cl4<O3\tl2h*)0`"
    none_type_1 = module_1.isclass(str_0)
    config_0.__getattr__(none_type_1)


def test_case_5():
    config_0 = module_0.Config()
    none_type_0 = config_0.update(**config_0)
    str_0 = "Cl4<O3\tl2h*)0`"
    bytes_0 = b";"
    none_type_1 = config_0.__setattr__(str_0, bytes_0)
    none_type_2 = config_0.__setattr__(str_0, str_0)
    config_0.__getattr__(config_0)


def test_case_6():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)


def test_case_7():
    none_type_0 = None
    config_0 = module_0.Config(
        none_type_0, none_type_0, none_type_0, converters=none_type_0
    )
    config_0.update_config(none_type_0)


def test_case_8():
    bool_0 = False
    none_type_0 = None
    config_0 = module_0.Config()
    config_0.__setitem__(none_type_0, bool_0)


def test_case_9():
    config_0 = module_0.Config()
    config_1 = module_0.Config(config_0)


def test_case_10():
    none_type_0 = None
    config_0 = module_0.Config(env_prefix=none_type_0)
    var_0 = config_0.load_environment_vars()
    var_1 = config_0.update_config(config_0)
    module_0.Config(env_prefix=config_0, converters=none_type_0)


def test_case_11():
    none_type_0 = None
    config_0 = module_0.Config(env_prefix=none_type_0)
    str_0 = "2\x0by[!b0nQ!"
    none_type_1 = config_0.__setattr__(str_0, none_type_0)
    config_0.update_config(str_0)


def test_case_12():
    none_type_0 = None
    config_0 = module_0.Config(env_prefix=none_type_0)
    var_0 = module_1.isclass(config_0)
    config_1 = module_0.Config(keep_alive=var_0, converters=config_0)
    list_0 = [config_0, config_0, config_0, config_1]
    module_0.Config(none_type_0, converters=list_0)
