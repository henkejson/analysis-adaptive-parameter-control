# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.config as module_0
import sanic.utils as module_1
import inspect as module_2


def test_case_0():
    config_0 = module_0.Config()


def test_case_1():
    str_0 = "r%;Ivc&d<)\x0bgW-w"
    config_0 = module_0.Config(converters=str_0)
    dict_0 = {str_0: str_0, str_0: str_0}
    str_1 = "kwHa\n,$A1\x0c?2"
    config_1 = module_0.Config()
    module_1.load_module_from_file_location(str_1, str_0, **dict_0)


def test_case_2():
    none_type_0 = None
    config_0 = module_0.Config(env_prefix=none_type_0)


def test_case_3():
    str_0 = "Y4(9\\O5\ro\x0bA$l~?E"
    dict_0 = {str_0: str_0}
    config_0 = module_0.Config(dict_0)
    module_2.getmembers(str_0, dict_0)


def test_case_4():
    config_0 = module_0.Config()
    none_type_0 = config_0.register_type(config_0)
    config_1 = module_2.getmembers(config_0)
    config_0.__getattr__(none_type_0)


def test_case_5():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)


def test_case_6():
    config_0 = module_0.Config()
    none_type_0 = config_0.register_type(config_0)


def test_case_7():
    config_0 = module_0.Config()
    config_1 = module_0.Config()
    none_type_0 = None
    config_0.__setitem__(config_1, none_type_0)


def test_case_8():
    float_0 = -307.338674
    module_0.Config(env_prefix=float_0)


def test_case_9():
    config_0 = module_0.Config()
    none_type_0 = None
    config_0.update_config(none_type_0)


def test_case_10():
    str_0 = "8bg'\"oqVsL"
    config_0 = module_0.Config(env_prefix=str_0)
    config_0.update_config(str_0)


def test_case_11():
    str_0 = "8bg'\"oqVsL"
    config_0 = module_0.Config(env_prefix=str_0)
    var_0 = config_0.load_environment_vars()
    config_1 = module_0.Config(env_prefix=str_0)
    var_1 = module_0.Config(env_prefix=var_0)
    config_0.__getattr__(config_1)


def test_case_12():
    none_type_0 = None
    str_0 = "_INIT"
    config_0 = module_0.Config(keep_alive=str_0)
    config_0.__setitem__(none_type_0, none_type_0)


def test_case_13():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)
    none_type_0 = config_0.update()
    none_type_1 = config_0.register_type(var_0)
    config_0.register_type(var_0)
