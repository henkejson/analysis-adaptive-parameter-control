# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.config as module_0
import inspect as module_1


def test_case_0():
    config_0 = module_0.Config()


def test_case_1():
    str_0 = "~`hG&5BK*m\x0c<UNrOv;L"
    str_1 = "Msp"
    str_2 = "v"
    dict_0 = {str_0: str_0, str_1: str_1, str_2: str_1, str_0: str_0}
    config_0 = module_0.Config(dict_0, converters=str_1)


def test_case_2():
    float_0 = 969.1462612001501
    list_0 = [float_0, float_0, float_0]
    none_type_0 = None
    config_0 = module_0.Config(keep_alive=list_0, converters=none_type_0)


def test_case_3():
    none_type_0 = None
    var_0 = module_1.isdatadescriptor(none_type_0)
    config_0 = module_0.Config(env_prefix=var_0, keep_alive=none_type_0)
    bool_0 = False
    bool_0.load_environment_vars()


def test_case_4():
    none_type_0 = None
    config_0 = module_0.Config(converters=none_type_0)
    config_0.update_config(none_type_0)


def test_case_5():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)


def test_case_6():
    str_0 = '"y+LU(%)3R^Ur_0'
    module_0.Config(keep_alive=str_0, converters=str_0)


def test_case_7():
    none_type_0 = None
    config_0 = module_0.Config(none_type_0, none_type_0, converters=none_type_0)
    str_0 = 'h3=":'
    bool_0 = True
    none_type_1 = config_0.__setitem__(str_0, bool_0)
    config_0.update_config(none_type_0)


def test_case_8():
    none_type_0 = None
    config_0 = module_0.Config(converters=none_type_0)
    none_type_1 = config_0.update()
    config_0.__getattr__(config_0)


def test_case_9():
    config_0 = module_0.Config()
    config_1 = module_0.Config(config_0)


def test_case_10():
    float_0 = 969.476655
    list_0 = [float_0, float_0, float_0]
    none_type_0 = None
    config_0 = module_0.Config(keep_alive=list_0, converters=none_type_0)
    none_type_1 = config_0.update(**config_0)
    config_0.__getattr__(none_type_0)


def test_case_11():
    str_0 = "\rYk"
    config_0 = module_0.Config(env_prefix=str_0)
    none_type_0 = config_0.register_type(config_0)


def test_case_12():
    none_type_0 = None
    config_0 = module_0.Config(converters=none_type_0)
    var_0 = config_0.update_config(config_0)
    str_0 = config_0.__str__()
    config_0.update_config(str_0)


def test_case_13():
    str_0 = "LOCAL_CERT_CREATOR"
    none_type_0 = None
    config_0 = module_0.Config(env_prefix=none_type_0)
    config_0.__setattr__(str_0, str_0)
