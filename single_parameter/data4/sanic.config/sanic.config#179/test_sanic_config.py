# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.config as module_0
import inspect as module_1


def test_case_0():
    config_0 = module_0.Config()


def test_case_1():
    str_0 = "A.Y$"
    config_0 = module_0.Config(converters=str_0)
    none_type_0 = config_0.__setattr__(str_0, str_0)


def test_case_2():
    str_0 = "O\n\"'Q=X\n"
    int_0 = 457
    dict_0 = {str_0: str_0, int_0: int_0, int_0: int_0}
    bool_0 = True
    str_1 = "+\r:D"
    dict_1 = {str_0: str_0, str_0: bool_0, str_1: str_0, str_1: bool_0}
    config_0 = module_0.Config(dict_1, keep_alive=bool_0)
    config_0.__getattr__(dict_0)


def test_case_3():
    bool_0 = False
    var_0 = module_1.isdatadescriptor(bool_0)
    config_0 = module_0.Config(env_prefix=var_0)


def test_case_4():
    str_0 = "parsed_cookies"
    str_1 = "="
    dict_0 = {str_1: str_1}
    config_0 = module_0.Config(dict_0)
    none_type_0 = config_0.__setitem__(str_0, str_0)


def test_case_5():
    none_type_0 = None
    config_0 = module_0.Config(keep_alive=none_type_0)
    config_0.__getattr__(none_type_0)


def test_case_6():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)


def test_case_7():
    config_0 = module_0.Config()
    none_type_0 = None
    config_0.update_config(none_type_0)


def test_case_8():
    bytes_0 = b"\x8c\x04\x84\xa0\xb7O"
    none_type_0 = None
    config_0 = module_0.Config(converters=none_type_0)
    none_type_1 = config_0.register_type(bytes_0)
    config_0.update_config(bytes_0)


def test_case_9():
    str_0 = "\x0c pr;O"
    str_1 = '\x0bB}\rUJ}h"gx=o+d\tC'
    config_0 = module_0.Config(env_prefix=str_1)
    none_type_0 = config_0.__setitem__(str_0, str_0)
    bool_0 = True
    set_0 = {bool_0, bool_0, bool_0, bool_0}
    var_0 = module_1.isclass(set_0)
    var_0.load_environment_vars()


def test_case_10():
    config_0 = module_0.Config()
    str_0 = "x\rCE.f~l>bnY"
    none_type_0 = None
    none_type_1 = config_0.__setattr__(str_0, none_type_0)
    var_0 = config_0.load_environment_vars()
    none_type_2 = config_0.register_type(none_type_1)
    config_0.register_type(var_0)


def test_case_11():
    config_0 = module_0.Config()
    var_0 = config_0.load_environment_vars()
    var_1 = config_0.update_config(config_0)
    str_0 = "LOCAL_CERT_CREATOR"
    config_0.__setattr__(str_0, str_0)
