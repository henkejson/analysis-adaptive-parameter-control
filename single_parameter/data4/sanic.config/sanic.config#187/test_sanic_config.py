# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.config as module_0


def test_case_0():
    config_0 = module_0.Config()


def test_case_1():
    str_0 = "L"
    module_0.Config(env_prefix=str_0, converters=str_0)


def test_case_2():
    str_0 = "L"
    config_0 = module_0.Config(env_prefix=str_0)


def test_case_3():
    none_type_0 = None
    config_0 = module_0.Config(env_prefix=none_type_0)
    var_0 = config_0.load_environment_vars()
    config_0.__getattr__(var_0)


def test_case_4():
    dict_0 = {}
    config_0 = module_0.Config(dict_0)
    config_0.__getattr__(dict_0)


def test_case_5():
    int_0 = -1213
    config_0 = module_0.Config(keep_alive=int_0)
    config_0.update_config(int_0)


def test_case_6():
    str_0 = "Co~70own'GR9tM|rjr"
    none_type_0 = None
    config_0 = module_0.Config(env_prefix=none_type_0)
    config_0.update_config(str_0)


def test_case_7():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)


def test_case_8():
    none_type_0 = None
    config_0 = module_0.Config(env_prefix=none_type_0)
    config_1 = module_0.Config()
    config_0.__getattr__(config_1)


def test_case_9():
    str_0 = "51<CWJrMU]v"
    config_0 = module_0.Config()
    str_1 = "\tm"
    str_2 = "0C"
    dict_0 = {str_0: str_0, str_1: str_0, str_2: str_1}
    config_1 = module_0.Config(dict_0, str_1, dict_0, converters=dict_0)
    var_0 = config_1.__getattr__(str_0)
    none_type_0 = None
    config_2 = module_0.Config(env_prefix=str_0, converters=none_type_0)
    config_2.__setattr__(none_type_0, config_2)


def test_case_10():
    str_0 = "51<CWJrMU]v"
    none_type_0 = None
    config_0 = module_0.Config(env_prefix=str_0, converters=none_type_0)
    none_type_1 = config_0.register_type(none_type_0)
    none_type_2 = config_0.__setattr__(str_0, config_0)
    str_1 = "\x0bR 3`LziT_kYM("
    none_type_3 = config_0.__setattr__(str_1, config_0)
    config_0.register_type(none_type_1)
