# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.config as module_0
import inspect as module_1


def test_case_0():
    config_0 = module_0.Config()


def test_case_1():
    str_0 = '#%Zjr-"b\x0cdnd\x0c0U;Dr'
    module_0.Config(converters=str_0)


def test_case_2():
    config_0 = module_0.Config()
    none_type_0 = None
    config_1 = module_0.Config(config_0, none_type_0, config_0)
    list_0 = [config_0]
    str_0 = "V!B"
    none_type_1 = config_0.__setitem__(str_0, config_0)
    var_0 = config_0.update_config(config_0)
    var_0.update(*list_0)


def test_case_3():
    str_0 = 'wzg^)3o/_>4d"p{>*'
    config_0 = module_0.Config(env_prefix=str_0)


def test_case_4():
    config_0 = module_0.Config()
    none_type_0 = None
    config_0.__getattr__(none_type_0)


def test_case_5():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)


def test_case_6():
    str_0 = "';V8'R^d"
    config_0 = module_0.Config()
    config_0.update_config(str_0)


def test_case_7():
    config_0 = module_0.Config()
    list_0 = [config_0]
    str_0 = "VDB"
    none_type_0 = config_0.__setitem__(str_0, config_0)
    var_0 = config_0.update_config(config_0)
    var_0.update(*list_0)


def test_case_8():
    none_type_0 = None
    config_0 = module_0.Config(none_type_0, none_type_0, converters=none_type_0)
    int_0 = 107
    config_0.__getattr__(int_0)


def test_case_9():
    none_type_0 = None
    str_0 = "';V8'R^d"
    module_0.Config(str_0, none_type_0)


def test_case_10():
    config_0 = module_0.Config()
    var_0 = module_1.isdatadescriptor(config_0)
    config_0.update_config(var_0)


def test_case_11():
    config_0 = module_0.Config()
    config_0.__getattr__(config_0)


def test_case_12():
    none_type_0 = None
    config_0 = module_0.Config(env_prefix=none_type_0)
    config_1 = module_0.Config(none_type_0)
    config_2 = module_0.Config(converters=config_0)
