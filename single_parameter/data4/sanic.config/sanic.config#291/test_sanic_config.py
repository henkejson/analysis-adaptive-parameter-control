# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.config as module_0
import sanic.errorpages as module_1
import inspect as module_2
import sanic.constants as module_3
import sanic.http.stream as module_4


def test_case_0():
    config_0 = module_0.Config()


def test_case_1():
    config_0 = module_0.Config()
    str_0 = "encode"
    str_1 = "uvU_&p}s]BTjf:39M\tZ"
    module_0.Config(keep_alive=str_1, converters=str_0)


def test_case_2():
    bool_0 = False
    config_0 = module_0.Config(keep_alive=bool_0)
    bytes_0 = b"\xad\xa5pXO5H\x1f\xfd\xb4a"
    config_1 = module_0.Config(keep_alive=bytes_0)
    var_0 = config_1.load_environment_vars()
    config_0.update(*var_0, **config_1)


def test_case_3():
    tuple_0 = ()
    config_0 = module_0.Config(env_prefix=tuple_0, converters=tuple_0)
    module_1.check_error_format(config_0)


def test_case_4():
    str_0 = "sY<Qg L]@F'Dr`}%dg?"
    module_0.Config(str_0)


def test_case_5():
    config_0 = module_0.Config()
    var_0 = module_2.getmembers(config_0)
    config_0.update_config(var_0)


def test_case_6():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)
    config_0.__getattr__(config_0)


def test_case_7():
    none_type_0 = None
    config_0 = module_0.Config(none_type_0)
    none_type_1 = config_0.register_type(config_0)
    float_0 = -3055.702902011032
    config_0.update_config(float_0)


def test_case_8():
    config_0 = module_0.Config()
    str_0 = "wzMAm+[fM^\x0cCsc}XZJq"
    none_type_0 = config_0.__setitem__(str_0, config_0)
    var_0 = config_0.update_config(config_0)
    none_type_1 = config_0.update()
    config_0.update_config(var_0)


def test_case_9():
    local_cert_creator_0 = module_3.LocalCertCreator.MKCERT
    var_0 = module_2.isdatadescriptor(local_cert_creator_0)
    list_0 = [var_0]
    config_0 = module_0.Config(converters=list_0)
    var_1 = config_0.load_environment_vars()
    stream_0 = module_4.Stream()
    var_2 = module_2.isdatadescriptor(stream_0)
    tuple_0 = (var_2,)
    dict_0 = {tuple_0: tuple_0}
    var_2.__new__(var_2, var_2, dict_0, stream_0)


def test_case_10():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)


def test_case_11():
    config_0 = module_0.Config()
    none_type_0 = None
    config_0.update_config(none_type_0)


def test_case_12():
    config_0 = module_0.Config()
    var_0 = config_0.load_environment_vars()
    str_0 = "1"
    config_1 = module_0.Config(env_prefix=str_0)
    none_type_0 = None
    config_0.update(*none_type_0, **config_0)


def test_case_13():
    str_0 = "<2t8Da3GKI0~$TtU2 `("
    none_type_0 = None
    config_0 = module_0.Config(none_type_0, converters=none_type_0)
    config_0.update_config(str_0)


def test_case_14():
    config_0 = module_0.Config()
    str_0 = "LOCAL_CERT_CREATOR"
    config_0.__setattr__(str_0, config_0)
