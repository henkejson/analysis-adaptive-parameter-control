# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.config as module_0
import inspect as module_1


def test_case_0():
    config_0 = module_0.Config()


def test_case_1():
    none_type_0 = None
    config_0 = module_0.Config(keep_alive=none_type_0)
    none_type_1 = None
    config_1 = module_0.Config(env_prefix=none_type_1)
    config_1.__getattr__(none_type_0)


def test_case_2():
    none_type_0 = None
    config_0 = module_0.Config(converters=none_type_0)
    config_0.__getattr__(config_0)


def test_case_3():
    bool_0 = False
    str_0 = "u,u-'"
    config_0 = module_0.Config(bool_0)
    config_0.update_config(str_0)


def test_case_4():
    none_type_0 = None
    config_0 = module_0.Config(keep_alive=none_type_0)
    config_1 = module_0.Config(converters=config_0)
    config_1.update_config(none_type_0)


def test_case_5():
    none_type_0 = None
    config_0 = module_0.Config(keep_alive=none_type_0)
    str_0 = "H-cYDM-wh[2KL\x0c@hX0d"
    config_1 = module_0.Config(env_prefix=str_0, keep_alive=str_0)
    config_1.update_config(none_type_0)


def test_case_6():
    none_type_0 = None
    config_0 = module_0.Config(converters=none_type_0)
    config_0.__getattr__(none_type_0)


def test_case_7():
    none_type_0 = None
    config_0 = module_0.Config(converters=none_type_0)
    config_0.update_config(none_type_0)


def test_case_8():
    dict_0 = {}
    none_type_0 = None
    config_0 = module_0.Config(converters=none_type_0)
    str_0 = "e~,>Do~"
    none_type_1 = config_0.__setitem__(str_0, none_type_0)
    config_0.__getattr__(dict_0)


def test_case_9():
    none_type_0 = None
    config_0 = module_0.Config(keep_alive=none_type_0)
    none_type_1 = None
    var_0 = module_1.getmembers(none_type_0)
    config_1 = module_0.Config(keep_alive=var_0)
    var_1 = module_1.getmembers(none_type_1)
    var_1.__getattr__(config_0)


def test_case_10():
    float_0 = -1084.577945
    module_0.Config(float_0, float_0)


def test_case_11():
    config_0 = module_0.Config()
    none_type_0 = config_0.update(**config_0)


def test_case_12():
    none_type_0 = None
    config_0 = module_0.Config(converters=none_type_0)
    var_0 = config_0.update_config(config_0)


def test_case_13():
    none_type_0 = None
    config_0 = module_0.Config(keep_alive=none_type_0)
    var_0 = config_0.update_config(config_0)
    none_type_1 = config_0.register_type(var_0)
    config_0.register_type(var_0)


def test_case_14():
    str_0 = ""
    config_0 = module_0.Config()
    config_0.load_environment_vars(str_0)
