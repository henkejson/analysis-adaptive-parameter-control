# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.config as module_0


def test_case_0():
    config_0 = module_0.Config()


def test_case_1():
    bool_0 = False
    config_0 = module_0.Config(keep_alive=bool_0)
    config_0.update_config(bool_0)


def test_case_2():
    str_0 = "&,"
    config_0 = module_0.Config(env_prefix=str_0)


def test_case_3():
    str_0 = "&,"
    config_0 = module_0.Config(env_prefix=str_0)
    str_1 = "MNmA\x0c9\x0bg:_~wH"
    config_0.__getattr__(str_1)


def test_case_4():
    str_0 = "L"
    none_type_0 = None
    config_0 = module_0.Config(env_prefix=str_0, keep_alive=none_type_0)
    config_0.update_config(str_0)


def test_case_5():
    str_0 = "L"
    config_0 = module_0.Config(env_prefix=str_0)


def test_case_6():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)


def test_case_7():
    str_0 = "L"
    config_0 = module_0.Config(converters=str_0)
    config_1 = module_0.Config(env_prefix=str_0)


def test_case_8():
    str_0 = "tV^\x0b)9!1#<\rlwx5"
    str_1 = 's`FR"B@t_'
    dict_0 = {str_0: str_0, str_0: str_0, str_1: str_1}
    config_0 = module_0.Config(dict_0)
    str_2 = "&,"
    config_1 = module_0.Config(env_prefix=str_2)


def test_case_9():
    none_type_0 = None
    config_0 = module_0.Config()
    config_1 = module_0.Config(
        none_type_0, none_type_0, none_type_0, converters=none_type_0
    )
    var_0 = config_0.update_config(config_0)
    var_1 = config_1.update_config(config_0)
    var_0.init_for_request()


def test_case_10():
    config_0 = module_0.Config()
    config_0.__getattr__(config_0)


def test_case_11():
    config_0 = module_0.Config()
    none_type_0 = config_0.update()
    var_0 = config_0.update_config(config_0)
    none_type_1 = config_0.register_type(var_0)
    config_0.register_type(var_0)


def test_case_12():
    str_0 = "L"
    bool_0 = False
    config_0 = module_0.Config(keep_alive=bool_0)
    module_0.Config(env_prefix=str_0, converters=str_0)


def test_case_13():
    str_0 = "_"
    bool_0 = False
    config_0 = module_0.Config(keep_alive=bool_0)
    config_1 = module_0.Config(env_prefix=str_0, converters=str_0)
    config_1.load_environment_vars(bool_0)
