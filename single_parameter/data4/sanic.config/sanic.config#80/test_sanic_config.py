# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.config as module_0
import sanic.http.stream as module_1
import inspect as module_2


def test_case_0():
    config_0 = module_0.Config()


def test_case_1():
    config_0 = module_0.Config()
    none_type_0 = config_0.register_type(config_0)
    var_0 = config_0.update_config(config_0)
    none_type_1 = None
    config_1 = module_0.Config(env_prefix=none_type_1, converters=config_0)
    config_0.update_config(none_type_1)


def test_case_2():
    bytes_0 = b"Proxy Authentication Required"
    config_0 = module_0.Config(keep_alive=bytes_0)


def test_case_3():
    bool_0 = True
    module_0.Config(env_prefix=bool_0)


def test_case_4():
    str_0 = ",^?QVg2k\tI^P,L!s85-u"
    str_1 = "0"
    set_0 = {str_1, str_0}
    str_2 = "Sd*%N!'\t,;t}BO:X\x0bR"
    dict_0 = {str_0: str_0, str_0: str_0, str_1: set_0, str_2: str_0}
    str_3 = "RM\nLFy[\r'VkB"
    dict_1 = {str_3: str_3, str_3: str_3}
    bool_0 = True
    config_0 = module_0.Config(dict_1, keep_alive=bool_0)
    none_type_0 = None
    none_type_1 = config_0.register_type(none_type_0)
    none_type_2 = config_0.update(**dict_0)


def test_case_5():
    config_0 = module_0.Config()
    stream_0 = module_1.Stream()
    config_0.register(stream_0, config_0)


def test_case_6():
    str_0 = "Cannot call recv_streaming while another task is already waiting for the next message"
    module_0.Config(converters=str_0)


def test_case_7():
    config_0 = module_0.Config()
    str_0 = "$\n;.}lfcs"
    none_type_0 = config_0.__setitem__(str_0, config_0)
    str_1 = "http.lifecycle.send"
    none_type_1 = config_0.register_type(none_type_0)
    var_0 = config_0.load_environment_vars()
    var_1 = config_0.load_environment_vars(str_1)
    none_type_2 = None
    config_0.update(**none_type_2)


def test_case_8():
    none_type_0 = None
    config_0 = module_0.Config(env_prefix=none_type_0, converters=none_type_0)
    str_0 = "d\\GhUY1~A"
    none_type_1 = config_0.__setattr__(str_0, none_type_0)
    var_0 = module_2.isclass(none_type_0)
    str_1 = "YvLy]ei&\\tD+og"
    config_0.update_config(str_1)


def test_case_9():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)


def test_case_10():
    config_0 = module_0.Config()
    none_type_0 = None
    config_0.update_config(none_type_0)


def test_case_11():
    bool_0 = False
    config_0 = module_0.Config(env_prefix=bool_0)


def test_case_12():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)
    bool_0 = False
    config_1 = module_0.Config(var_0, var_0, bool_0)
    config_0.__getattr__(config_1)


def test_case_13():
    config_0 = module_0.Config()
    var_0 = module_2.getmembers(config_0)
    stream_0 = module_1.Stream()
    str_0 = "LOCAL_CERT_CREATOR"
    config_0.__setattr__(str_0, str_0)
