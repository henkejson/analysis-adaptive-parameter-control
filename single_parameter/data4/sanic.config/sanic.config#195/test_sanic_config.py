# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.config as module_0
import abc as module_1
import inspect as module_2


def test_case_0():
    config_0 = module_0.Config()


def test_case_1():
    str_0 = "transfer-encoding"
    bytes_0 = b"M9e\xc0\r["
    config_0 = module_0.Config(env_prefix=str_0, converters=bytes_0)
    str_0.update(*bytes_0, **str_0)


def test_case_2():
    complex_0 = -3217.216 - 1568.5j
    none_type_0 = None
    config_0 = module_0.Config(keep_alive=complex_0, converters=none_type_0)
    module_1.ABCMeta()


def test_case_3():
    none_type_0 = None
    config_0 = module_0.Config(none_type_0, none_type_0)


def test_case_4():
    str_0 = ""
    float_0 = 3628.037287200777
    str_1 = "ftDt3"
    dict_0 = {str_1: str_0}
    none_type_0 = None
    config_0 = module_0.Config(dict_0, str_0, converters=none_type_0)
    var_0 = module_2.getmembers(float_0)
    var_0.__setitem__(str_0, float_0)


def test_case_5():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)


def test_case_6():
    str_0 = '6`=F/iA"~'
    config_0 = module_0.Config(env_prefix=str_0, keep_alive=str_0)
    config_0.update_config(str_0)


def test_case_7():
    str_0 = ";G`kt&A{PAwrbQuW"
    module_0.Config(env_prefix=str_0, converters=str_0)


def test_case_8():
    config_0 = module_0.Config()
    var_0 = module_2.getmembers(config_0)
    none_type_0 = config_0.register_type(config_0)
    var_1 = module_2.isclass(var_0)
    var_1.__setitem__(var_0, var_0)


def test_case_9():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)
    config_0.__setitem__(var_0, var_0)


def test_case_10():
    config_0 = module_0.Config()
    var_0 = module_2.getmembers(config_0)


def test_case_11():
    int_0 = -3089
    config_0 = module_0.Config()
    config_0.update_config(int_0)


def test_case_12():
    config_0 = module_0.Config()
    config_0.__getattr__(config_0)
