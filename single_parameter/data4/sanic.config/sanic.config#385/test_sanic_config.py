# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.config as module_0
import inspect as module_1
import builtins as module_2


def test_case_0():
    config_0 = module_0.Config()


def test_case_1():
    str_0 = "UOG.oFg\x0cBiRy|<"
    config_0 = module_0.Config(env_prefix=str_0)


def test_case_2():
    config_0 = module_0.Config()
    bool_0 = True
    config_0.__getattr__(bool_0)


def test_case_3():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)


def test_case_4():
    none_type_0 = None
    config_0 = module_0.Config(none_type_0)
    config_0.update_config(none_type_0)


def test_case_5():
    none_type_0 = None
    config_0 = module_0.Config(none_type_0)
    none_type_1 = config_0.register_type(config_0)


def test_case_6():
    none_type_0 = None
    config_0 = module_0.Config(none_type_0)
    str_0 = "X|#t$6 \x0c\x0c\tP$Nb"
    var_0 = module_1.getmembers(str_0)
    str_1 = ">RuY&2^=Pcy0k"
    none_type_1 = config_0.__setitem__(str_1, str_1)


def test_case_7():
    complex_0 = 777.8918 - 959j
    list_0 = [complex_0]
    none_type_0 = None
    config_0 = module_0.Config(
        none_type_0, none_type_0, none_type_0, converters=none_type_0
    )
    module_2.object(*list_0, **config_0)


def test_case_8():
    config_0 = module_0.Config()
    config_0.__getattr__(config_0)


def test_case_9():
    none_type_0 = None
    config_0 = module_0.Config(none_type_0)
    str_0 = "X|#tV$6 \x0c\x0c\tP$qNb"
    config_0.update_config(str_0)


def test_case_10():
    bool_0 = True
    module_0.Config(keep_alive=bool_0, converters=bool_0)


def test_case_11():
    config_0 = module_0.Config()
    bool_0 = False
    str_0 = "UOG.oFg\x0cBiRy|<"
    config_1 = module_0.Config(bool_0, converters=str_0)
    config_0.__getattr__(config_0)


def test_case_12():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)
    str_0 = "UOG.oFg\x0cBiRy|<"
    config_1 = module_0.Config(keep_alive=str_0)
    config_0.__getattr__(config_1)


def test_case_13():
    str_0 = "a\x0c8`d  f1{ks6US/?IK"
    none_type_0 = None
    module_0.Config(str_0, keep_alive=none_type_0)


def test_case_14():
    config_0 = module_0.Config()
    none_type_0 = config_0.register_type(config_0)
    config_0.register_type(config_0)


def test_case_15():
    config_0 = module_0.Config()
    str_0 = "Static route must be a valid path, not "
    none_type_0 = config_0.register_type(str_0)
    str_1 = "LOCAL_CERT_CREATOR"
    config_0.__setattr__(str_1, str_1)
