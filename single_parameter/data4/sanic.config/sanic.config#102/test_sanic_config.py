# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.config as module_0


def test_case_0():
    config_0 = module_0.Config()


def test_case_1():
    str_0 = '<AV,"\rv'
    int_0 = 36
    module_0.Config(env_prefix=str_0, converters=int_0)


def test_case_2():
    str_0 = '<AV,"\rv'
    int_0 = 36
    module_0.Config(env_prefix=int_0, keep_alive=str_0, converters=str_0)


def test_case_3():
    none_type_0 = None
    config_0 = module_0.Config(env_prefix=none_type_0, keep_alive=none_type_0)


def test_case_4():
    str_0 = "I7pR"
    bytes_0 = b"Request-URI Too Long"
    config_0 = module_0.Config(env_prefix=str_0, keep_alive=bytes_0, converters=str_0)
    config_0.__getattr__(bytes_0)


def test_case_5():
    none_type_0 = None
    config_0 = module_0.Config(none_type_0, none_type_0)
    none_type_1 = config_0.register_type(none_type_0)


def test_case_6():
    tuple_0 = ()
    tuple_1 = (tuple_0,)
    list_0 = [tuple_1, tuple_1, tuple_0, tuple_0]
    config_0 = module_0.Config(converters=tuple_1)
    config_0.__setitem__(tuple_1, list_0)


def test_case_7():
    bool_0 = True
    str_0 = "h\r`g_p^gDp*$+T/}"
    none_type_0 = None
    config_0 = module_0.Config(env_prefix=str_0, converters=none_type_0)
    config_0.update(**bool_0)


def test_case_8():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)


def test_case_9():
    int_0 = -383
    none_type_0 = None
    module_0.Config(int_0, keep_alive=none_type_0)


def test_case_10():
    none_type_0 = None
    config_0 = module_0.Config(none_type_0, none_type_0)
    config_0.update_config(none_type_0)


def test_case_11():
    str_0 = "e?:/qng\\Hco"
    config_0 = module_0.Config(env_prefix=str_0)
    config_0.update_config(str_0)


def test_case_12():
    bytes_0 = b"Request-URI Too Long"
    module_0.Config(env_prefix=bytes_0, keep_alive=bytes_0, converters=bytes_0)


def test_case_13():
    str_0 = "\tOpR"
    config_0 = module_0.Config(env_prefix=str_0)
    bytes_0 = b"Request-URI Too Long"
    config_1 = module_0.Config(env_prefix=str_0, keep_alive=bytes_0, converters=str_0)
    config_0.__getattr__(config_0)


def test_case_14():
    none_type_0 = None
    config_0 = module_0.Config(keep_alive=none_type_0)
    str_0 = "LOCAL_CERT_CREATOR"
    config_0.__setitem__(str_0, str_0)
