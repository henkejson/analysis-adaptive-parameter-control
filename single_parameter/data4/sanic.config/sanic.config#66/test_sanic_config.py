# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.config as module_0


def test_case_0():
    config_0 = module_0.Config()


def test_case_1():
    str_0 = "#"
    config_0 = module_0.Config(env_prefix=str_0, converters=str_0)


def test_case_2():
    str_0 = "S"
    module_0.Config(env_prefix=str_0, keep_alive=str_0, converters=str_0)


def test_case_3():
    none_type_0 = None
    config_0 = module_0.Config(env_prefix=none_type_0, keep_alive=none_type_0)


def test_case_4():
    str_0 = "#"
    config_0 = module_0.Config(env_prefix=str_0, converters=str_0)
    var_0 = config_0.load_environment_vars()
    config_1 = module_0.Config(config_0, keep_alive=config_0)
    none_type_0 = None
    config_0.update_config(none_type_0)


def test_case_5():
    str_0 = "#"
    config_0 = module_0.Config(env_prefix=str_0, converters=str_0)
    config_0.log_response()


def test_case_6():
    none_type_0 = None
    config_0 = module_0.Config(none_type_0, none_type_0, none_type_0)
    var_0 = config_0.update_config(config_0)
    none_type_1 = config_0.register_type(none_type_0)
    list_0 = []
    config_0.__getattr__(list_0)


def test_case_7():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)


def test_case_8():
    str_0 = "S"
    none_type_0 = None
    config_0 = module_0.Config(
        env_prefix=str_0, keep_alive=str_0, converters=none_type_0
    )


def test_case_9():
    str_0 = "\r\r"
    module_0.Config(env_prefix=str_0, converters=str_0)


def test_case_10():
    config_0 = module_0.Config()
    none_type_0 = None
    config_0.update_config(none_type_0)


def test_case_11():
    bytes_0 = b"See Other"
    str_0 = "#"
    config_0 = module_0.Config(env_prefix=str_0, converters=str_0)
    config_0.update_config(bytes_0)


def test_case_12():
    str_0 = "#?"
    str_1 = ":<"
    str_2 = "*CC-Gu"
    str_3 = "_FALLBACK_ERROR_FORMAT"
    dict_0 = {str_1: str_0, str_2: str_2, str_2: str_2, str_3: str_0}
    module_0.Config(dict_0)


def test_case_13():
    str_0 = "_"
    none_type_0 = None
    config_0 = module_0.Config(
        env_prefix=str_0, keep_alive=str_0, converters=none_type_0
    )
    var_0 = config_0.update_config(config_0)
    config_0.update_config(var_0)
