# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.config as module_0
import sanic.utils as module_1


def test_case_0():
    config_0 = module_0.Config()


def test_case_1():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)
    var_1 = config_0.load_environment_vars()
    var_2 = module_0.Config(converters=config_0)


def test_case_2():
    none_type_0 = None
    config_0 = module_0.Config(env_prefix=none_type_0)
    bytes_0 = b"Created"
    module_1.load_module_from_file_location(bytes_0, bytes_0)


def test_case_3():
    float_0 = 1617.0
    none_type_0 = None
    config_0 = module_0.Config(none_type_0, none_type_0, none_type_0)
    config_0.__getattr__(float_0)


def test_case_4():
    config_0 = module_0.Config()
    config_0.__getattr__(config_0)


def test_case_5():
    complex_0 = 141.4031 - 172j
    list_0 = [complex_0, complex_0, complex_0]
    module_0.Config(converters=list_0)


def test_case_6():
    config_0 = module_0.Config()
    none_type_0 = None
    config_0.update_config(none_type_0)


def test_case_7():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)


def test_case_8():
    config_0 = module_0.Config()
    str_0 = "pi1'O("
    bool_0 = False
    config_1 = module_0.Config(keep_alive=bool_0)
    config_0.__getattr__(str_0)


def test_case_9():
    float_0 = -754.5
    module_0.Config(float_0)


def test_case_10():
    config_0 = module_0.Config()
    str_0 = "X3VA)1"
    none_type_0 = config_0.__setitem__(str_0, config_0)
    none_type_1 = None
    config_0.update_config(none_type_1)


def test_case_11():
    str_0 = "MR0*\x0bN=6-_"
    config_0 = module_0.Config(env_prefix=str_0)
    var_0 = config_0.load_environment_vars(str_0)
    var_0.update_config(str_0)


def test_case_12():
    str_0 = "[:@9"
    config_0 = module_0.Config()
    none_type_0 = None
    none_type_1 = config_0.register_type(none_type_0)
    var_0 = config_0.load_environment_vars(str_0)
    config_0.update_config(str_0)


def test_case_13():
    config_0 = module_0.Config()
    str_0 = "LOCAL_CERT_CREATOR"
    config_0.__setattr__(str_0, config_0)
