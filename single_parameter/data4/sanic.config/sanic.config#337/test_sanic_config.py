# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.config as module_0
import inspect as module_1


def test_case_0():
    bytes_0 = b"J}}a\x92\x1d\x91\x17\x15\xfc\x1e\xc8\x87s\x18\x19a\x14"
    module_0.DescriptorMeta(bytes_0)


def test_case_1():
    str_0 = "_INIT"
    config_0 = module_0.Config(env_prefix=str_0)


def test_case_2():
    bool_0 = False
    none_type_0 = None
    config_0 = module_0.Config(keep_alive=bool_0, converters=none_type_0)


def test_case_3():
    none_type_0 = None
    config_0 = module_0.Config(converters=none_type_0)


def test_case_4():
    bool_0 = True
    dict_0 = {bool_0: bool_0, bool_0: bool_0, bool_0: bool_0, bool_0: bool_0}
    config_0 = module_0.Config(dict_0)


def test_case_5():
    none_type_0 = None
    config_0 = module_0.Config(converters=none_type_0)
    none_type_1 = config_0.update(**config_0)
    var_0 = config_0.update_config(config_0)


def test_case_6():
    bool_0 = False
    none_type_0 = None
    config_0 = module_0.Config(keep_alive=bool_0, converters=none_type_0)
    config_0.update_config(bool_0)


def test_case_7():
    bool_0 = False
    config_0 = module_0.Config(keep_alive=bool_0, converters=bool_0)
    config_1 = module_0.Config(converters=config_0)
    none_type_0 = config_0.update(**config_1)
    module_0.DescriptorMeta(none_type_0)


def test_case_8():
    none_type_0 = None
    config_0 = module_0.Config(env_prefix=none_type_0)
    config_0.__bool__()


def test_case_9():
    bool_0 = False
    config_0 = module_0.Config(keep_alive=bool_0, converters=bool_0)
    config_0.__getattr__(config_0)


def test_case_10():
    bool_0 = False
    config_0 = module_0.Config(keep_alive=bool_0, converters=bool_0)
    str_0 = "o8sc"
    none_type_0 = config_0.__setitem__(str_0, bool_0)
    config_1 = module_0.Config(converters=config_0)
    none_type_1 = config_0.update(**config_1)
    config_0.update_config(bool_0)


def test_case_11():
    bool_0 = False
    config_0 = module_0.Config(keep_alive=bool_0, converters=bool_0)
    config_1 = module_0.Config(converters=config_0)
    str_0 = "TqtWf]m}K"
    dict_0 = {str_0: str_0}
    none_type_0 = config_0.update(**dict_0)
    str_1 = 'HGqQw^"1'
    config_0.update_config(str_1)


def test_case_12():
    bool_0 = False
    var_0 = module_1.getmembers(bool_0)
    bytes_0 = b"\x81\x95\xad\xbf-\x15\xad\xf4\xe5"
    module_0.Config(converters=bytes_0)
