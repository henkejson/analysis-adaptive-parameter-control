# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.config as module_0
import inspect as module_1


def test_case_0():
    config_0 = module_0.Config()


def test_case_1():
    none_type_0 = None
    set_0 = {none_type_0}
    config_0 = module_0.Config(none_type_0, keep_alive=set_0)
    var_0 = config_0.load_environment_vars()


def test_case_2():
    none_type_0 = None
    config_0 = module_0.Config(env_prefix=none_type_0)
    none_type_1 = config_0.register_type(config_0)


def test_case_3():
    config_0 = module_0.Config()
    config_0.__getattr__(config_0)


def test_case_4():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)


def test_case_5():
    config_0 = module_0.Config()
    none_type_0 = None
    config_0.update_config(none_type_0)


def test_case_6():
    config_0 = module_0.Config()
    str_0 = "!64F7{ D9TgFff"
    none_type_0 = None
    none_type_1 = config_0.__setattr__(str_0, none_type_0)
    none_type_2 = config_0.register_type(config_0)
    var_0 = config_0.load_environment_vars()
    config_1 = module_0.Config(keep_alive=var_0, converters=var_0)
    config_1.__setitem__(none_type_2, config_0)


def test_case_7():
    config_0 = module_0.Config()
    none_type_0 = module_1.getmembers(config_0)


def test_case_8():
    none_type_0 = None
    config_0 = module_0.Config(env_prefix=none_type_0, converters=none_type_0)
    str_0 = "\nLogger used by Sanic for access logging\n"
    none_type_1 = config_0.__setattr__(str_0, none_type_0)
    config_0.update_config(str_0)


def test_case_9():
    config_0 = module_0.Config()
    config_1 = module_0.Config(converters=config_0)
    none_type_0 = config_0.register_type(config_1)
    none_type_1 = config_1.update()


def test_case_10():
    str_0 = "3w_m]'.=!#*Jh9>P\"GNE"
    str_1 = "_FALLBACK_ERROR_FORMAT"
    str_2 = "Z-$C&Q;fvDLHT:pC"
    none_type_0 = None
    config_0 = module_0.Config(none_type_0, str_2, none_type_0)
    float_0 = 20.0
    none_type_1 = config_0.register_type(float_0)
    dict_0 = {str_0: str_0, str_1: str_0, str_2: str_2, str_0: str_0}
    none_type_2 = None
    module_0.Config(dict_0, none_type_2, none_type_2)


def test_case_11():
    config_0 = module_0.Config()
    str_0 = "Jh"
    none_type_0 = config_0.__setitem__(str_0, str_0)
    var_0 = config_0.update_config(config_0)
    none_type_1 = config_0.register_type(var_0)
    config_1 = module_0.Config(env_prefix=var_0, converters=var_0)
    config_0.register_type(var_0)
