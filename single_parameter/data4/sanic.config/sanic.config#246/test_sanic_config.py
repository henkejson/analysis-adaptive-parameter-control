# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.config as module_0
import inspect as module_1
import sanic.constants as module_2


def test_case_0():
    config_0 = module_0.Config()


def test_case_1():
    bool_0 = False
    var_0 = module_1.isclass(bool_0)
    var_1 = var_0.__repr__()
    config_0 = module_0.Config(converters=var_1)
    config_1 = module_0.Config()
    var_2 = config_1.update_config(config_1)
    str_0 = "gw0\\_Dn\tP'C\nH"
    str_1 = "gw0\\_Dn\tP'C\nH"
    var_2.__setitem__(str_1, str_0)


def test_case_2():
    none_type_0 = None
    config_0 = module_0.Config(env_prefix=none_type_0)


def test_case_3():
    config_0 = module_0.Config()
    config_0.init_for_request()


def test_case_4():
    config_0 = module_0.Config()
    config_0.__getattr__(config_0)


def test_case_5():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)


def test_case_6():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)
    none_type_0 = config_0.register_type(var_0)
    str_0 = "gw0\\_Dn\tP'C\nH"
    var_0.__setitem__(str_0, var_0)


def test_case_7():
    config_0 = module_0.Config()
    config_0.__setitem__(config_0, config_0)


def test_case_8():
    none_type_0 = None
    str_0 = "proxy-authenticate"
    config_0 = module_0.Config(env_prefix=str_0)
    var_0 = module_1.isclass(none_type_0)
    module_0.DescriptorMeta(var_0)


def test_case_9():
    config_0 = module_0.Config()
    float_0 = 1356.84174
    var_0 = module_1.isclass(float_0)
    config_1 = module_0.Config(env_prefix=var_0, keep_alive=var_0)
    config_2 = module_0.Config()
    var_0.register_type(var_0)


def test_case_10():
    none_type_0 = None
    config_0 = module_0.Config(env_prefix=none_type_0)
    config_0.update_config(none_type_0)


def test_case_11():
    config_0 = module_0.Config()
    var_0 = module_2.LocalCertCreator.AUTO
    config_0.update_config(var_0)


def test_case_12():
    complex_0 = 1341.16 + 916.64496j
    module_0.Config(complex_0)


def test_case_13():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)
    none_type_0 = config_0.register_type(var_0)
    config_0.register_type(var_0)
