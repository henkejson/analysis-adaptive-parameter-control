# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.config as module_0
import inspect as module_1


def test_case_0():
    config_0 = module_0.Config()


def test_case_1():
    bytes_0 = b"\x8fy\x1d\x8a8K\xf3d\xaa\xa8\xc86\x98n"
    config_0 = module_0.Config(keep_alive=bytes_0, converters=bytes_0)
    var_0 = config_0.update_config(config_0)


def test_case_2():
    none_type_0 = None
    var_0 = module_1.isclass(none_type_0)
    config_0 = module_0.Config(env_prefix=none_type_0)


def test_case_3():
    str_0 = "b#-A@$]"
    dict_0 = {str_0: str_0}
    none_type_0 = None
    config_0 = module_0.Config(dict_0, converters=none_type_0)
    none_type_1 = config_0.register_type(none_type_0)
    none_type_2 = None
    module_0.Config(str_0, converters=none_type_2)


def test_case_4():
    config_0 = module_0.Config()
    config_0.__getattr__(config_0)


def test_case_5():
    config_0 = module_0.Config()
    none_type_0 = None
    config_0.update_config(none_type_0)


def test_case_6():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)


def test_case_7():
    bytes_0 = b"\x8fy\x1dP8K\xf3d\xaa\xa8\xc86\x98n"
    set_0 = {bytes_0, bytes_0}
    config_0 = module_0.Config(keep_alive=bytes_0, converters=bytes_0)
    var_0 = config_0.update_config(config_0)
    var_1 = module_1.isclass(config_0)
    config_0.__setitem__(set_0, var_0)


def test_case_8():
    str_0 = "td\tRD{Y1"
    str_1 = "d7$uYk!"
    config_0 = module_0.Config(env_prefix=str_0, keep_alive=str_1)
    config_0.__getattr__(config_0)


def test_case_9():
    str_0 = "application/json"
    config_0 = module_0.Config()
    config_0.__getattr__(str_0)


def test_case_10():
    config_0 = module_0.Config()
    none_type_0 = config_0.update(**config_0)


def test_case_11():
    bytes_0 = b"\xd0-\x1df\x18\xb3\x8b\x8a\x88d\xb2\x96\xb3\xb1U;i"
    module_0.Config(keep_alive=bytes_0, converters=bytes_0)


def test_case_12():
    bytes_0 = b"\x8fy\x1dP8K\xf3d\xaa\xa8\xc86\x98n"
    config_0 = module_0.Config()
    config_0.update_config(bytes_0)
