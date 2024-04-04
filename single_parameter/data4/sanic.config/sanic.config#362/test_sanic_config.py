# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.config as module_0
import inspect as module_1
import sanic.errorpages as module_2


def test_case_0():
    config_0 = module_0.Config()


def test_case_1():
    none_type_0 = None
    complex_0 = 3830.9639 + 2448.73314j
    none_type_1 = None
    module_0.Config(none_type_0, none_type_1, none_type_0, converters=complex_0)


def test_case_2():
    bool_0 = True
    none_type_0 = None
    config_0 = module_0.Config()
    none_type_1 = config_0.register_type(none_type_0)
    module_0.Config(config_0, keep_alive=none_type_0, converters=bool_0)


def test_case_3():
    none_type_0 = None
    config_0 = module_0.Config(none_type_0)
    config_0.update_config(none_type_0)


def test_case_4():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)


def test_case_5():
    str_0 = ">E&~q<ifV+Glq\\"
    none_type_0 = None
    module_0.Config(none_type_0, converters=str_0)


def test_case_6():
    str_0 = "JSONResponse"
    none_type_0 = None
    config_0 = module_0.Config(converters=none_type_0)
    none_type_1 = config_0.register_type(str_0)


def test_case_7():
    bool_0 = True
    dict_0 = module_1.isdatadescriptor(bool_0)
    config_0 = module_0.Config(env_prefix=dict_0, keep_alive=bool_0, converters=dict_0)
    int_0 = -1117
    module_0.Config(env_prefix=bool_0, keep_alive=int_0, converters=config_0)


def test_case_8():
    config_0 = module_0.Config()
    var_0 = module_1.getmembers(config_0)
    var_0.load_environment_vars(var_0)


def test_case_9():
    bool_0 = True
    none_type_0 = None
    config_0 = module_0.Config(none_type_0, keep_alive=bool_0)
    none_type_1 = config_0.register_type(none_type_0)
    none_type_2 = None
    config_1 = module_0.Config(converters=none_type_2)
    var_0 = config_0.update_config(config_0)


def test_case_10():
    none_type_0 = None
    config_0 = module_0.Config(env_prefix=none_type_0, converters=none_type_0)
    var_0 = config_0.update_config(config_0)


def test_case_11():
    str_0 = "JSONResponse"
    none_type_0 = None
    config_0 = module_0.Config(converters=none_type_0)
    none_type_1 = config_0.update(**config_0)
    none_type_2 = config_0.__setitem__(str_0, str_0)
    module_2.check_error_format(none_type_2)


def test_case_12():
    bool_0 = True
    none_type_0 = None
    config_0 = module_0.Config(converters=none_type_0)
    none_type_1 = config_0.register_type(bool_0)
    var_0 = config_0.update_config(config_0)
    config_0.__getattr__(config_0)


def test_case_13():
    bool_0 = True
    none_type_0 = None
    config_0 = module_0.Config(converters=none_type_0)
    none_type_1 = config_0.register_type(bool_0)
    config_1 = module_0.Config()
    str_0 = "V8!n*sx"
    config_1.update_config(str_0)


def test_case_14():
    bool_0 = True
    none_type_0 = None
    config_0 = module_0.Config(converters=none_type_0)
    none_type_1 = config_0.register_type(bool_0)
    str_0 = "LOCAL_CERT_CREATOR"
    bytes_0 = b"\x17\x12\x96\xa6\x90\xb7#|B\xeaJ9\x15\xd4"
    dict_0 = {str_0: bytes_0, str_0: bytes_0}
    config_0.update(**dict_0)


def test_case_15():
    bool_0 = False
    str_0 = "~$O@<dr%_\th1Ect\t"
    str_1 = "]jes3)0NSP<FPm+"
    str_2 = "_FALLBACK_ERROR_FORMAT"
    dict_0 = {str_0: bool_0, str_1: str_1, str_1: str_1, str_2: str_1}
    module_0.Config(dict_0, keep_alive=bool_0)
