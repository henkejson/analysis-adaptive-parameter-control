# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.config as module_0
import sanic.http.http1 as module_1
import inspect as module_2


def test_case_0():
    config_0 = module_0.Config()


def test_case_1():
    bool_0 = False
    config_0 = module_0.Config(env_prefix=bool_0, keep_alive=bool_0, converters=bool_0)
    config_1 = module_0.Config()
    var_0 = config_1.update_config(config_1)
    none_type_0 = config_0.register_type(var_0)
    var_1 = config_0.load_environment_vars()
    config_1.update_config(var_1)


def test_case_2():
    bytes_0 = b"0]"
    module_0.Config(env_prefix=bytes_0)


def test_case_3():
    config_0 = module_0.Config()
    config_1 = module_0.Config(config_0)
    var_0 = config_0.update_config(config_1)
    none_type_0 = config_1.register_type(var_0)
    var_1 = config_0.load_environment_vars()
    config_1.update_config(var_1)


def test_case_4():
    config_0 = module_0.Config()
    module_1.Http(config_0)


def test_case_5():
    bool_0 = False
    bool_1 = False
    config_0 = module_0.Config(env_prefix=bool_0, keep_alive=bool_0, converters=bool_1)
    config_1 = module_0.Config()
    bytes_0 = b"Expectation Failed"
    config_1.__setitem__(bytes_0, config_0)


def test_case_6():
    none_type_0 = None
    config_0 = module_0.Config(converters=none_type_0)
    str_0 = "h%>t\tH:{'UOg^}"
    none_type_1 = config_0.__setitem__(str_0, config_0)
    int_0 = 2782
    module_0.Config(env_prefix=str_0, keep_alive=str_0, converters=int_0)


def test_case_7():
    dict_0 = {}
    none_type_0 = None
    config_0 = module_0.Config(dict_0, none_type_0)
    config_0.__getattr__(dict_0)


def test_case_8():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)


def test_case_9():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)
    config_0.update_config(var_0)


def test_case_10():
    bytes_0 = b"%x\r\n%b\r\n"
    none_type_0 = None
    config_0 = module_0.Config(converters=none_type_0)
    var_0 = config_0.update_config(config_0)
    var_1 = module_2.isdatadescriptor(bytes_0)
    var_2 = module_2.isdatadescriptor(none_type_0)
    str_0 = "V}k<a#-c}%G%Lp<Id\x0cI"
    module_0.Config(converters=str_0)


def test_case_11():
    none_type_0 = None
    config_0 = module_0.Config(converters=none_type_0)
    none_type_1 = module_2.isdatadescriptor(config_0)
    config_1 = module_0.Config(none_type_0, keep_alive=none_type_1, converters=config_0)
    bytes_0 = b"Expectation Failed"
    config_0.__getattr__(bytes_0)


def test_case_12():
    bytes_0 = b"%x\r\n%b\r\n"
    none_type_0 = None
    config_0 = module_0.Config(converters=none_type_0)
    config_1 = module_0.Config()
    config_1.update_config(bytes_0)


def test_case_13():
    config_0 = module_0.Config()
    str_0 = ""
    config_0.load_environment_vars(str_0)


def test_case_14():
    bool_0 = False
    str_0 = "H"
    config_0 = module_0.Config(env_prefix=str_0, keep_alive=bool_0, converters=bool_0)
    config_0.__instancecheck__(bool_0, str_0)


def test_case_15():
    bool_0 = False
    config_0 = module_0.Config(env_prefix=bool_0, keep_alive=bool_0, converters=bool_0)
    str_0 = "H"
    module_0.Config(env_prefix=str_0, keep_alive=bool_0, converters=config_0)


def test_case_16():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)
    str_0 = "_"
    var_1 = config_0.load_environment_vars(str_0)
    config_0.update_config(str_0)
