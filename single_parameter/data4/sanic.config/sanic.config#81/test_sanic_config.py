# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.config as module_0


def test_case_0():
    config_0 = module_0.Config()


def test_case_1():
    str_0 = "%x}:X7wY!a|Vt>Y(26H@"
    none_type_0 = None
    dict_0 = {str_0: none_type_0}
    module_0.Config(dict_0, converters=str_0)


def test_case_2():
    bool_0 = False
    config_0 = module_0.Config(keep_alive=bool_0)


def test_case_3():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)


def test_case_4():
    config_0 = module_0.Config()
    none_type_0 = None
    config_0.__getattr__(none_type_0)


def test_case_5():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)
    config_0.__setitem__(var_0, var_0)


def test_case_6():
    config_0 = module_0.Config()
    none_type_0 = None
    config_0.update_config(none_type_0)


def test_case_7():
    str_0 = "."
    config_0 = module_0.Config(env_prefix=str_0)


def test_case_8():
    none_type_0 = None
    config_0 = module_0.Config(none_type_0, none_type_0, converters=none_type_0)
    config_0.update_config(none_type_0)


def test_case_9():
    config_0 = module_0.Config()
    config_0.__getattr__(config_0)


def test_case_10():
    config_0 = module_0.Config()
    bool_0 = True
    config_1 = module_0.Config(keep_alive=bool_0, converters=config_0)
    none_type_0 = None
    var_0 = config_0.update_config(config_0)
    config_0.register(none_type_0, config_0)


def test_case_11():
    str_0 = "=g\x0cR\nt\rJxK~9.qH~gyaS"
    config_0 = module_0.Config()
    config_0.update_config(str_0)
