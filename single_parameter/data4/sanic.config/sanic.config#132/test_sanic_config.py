# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.config as module_0
import inspect as module_1


def test_case_0():
    config_0 = module_0.Config()


def test_case_1():
    none_type_0 = None
    config_0 = module_0.Config(converters=none_type_0)
    config_1 = module_0.Config(keep_alive=config_0, converters=config_0)
    config_1.__setitem__(config_0, none_type_0)


def test_case_2():
    none_type_0 = None
    config_0 = module_0.Config(env_prefix=none_type_0, converters=none_type_0)
    none_type_1 = config_0.update(**config_0)


def test_case_3():
    none_type_0 = None
    config_0 = module_0.Config(converters=none_type_0)
    str_0 = "7d\nh$qrbG~\t\x0b#"
    config_0.__getattr__(str_0)


def test_case_4():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)


def test_case_5():
    none_type_0 = None
    config_0 = module_0.Config(converters=none_type_0)
    str_0 = "7d\nh$qrbG~\t\x0b#"
    none_type_1 = config_0.__setitem__(str_0, config_0)
    var_0 = config_0.update_config(config_0)


def test_case_6():
    str_0 = "breadcrumbs"
    dict_0 = {str_0: str_0}
    config_0 = module_0.Config(dict_0, str_0, str_0)


def test_case_7():
    none_type_0 = None
    config_0 = module_0.Config()
    config_0.update_config(none_type_0)


def test_case_8():
    none_type_0 = None
    config_0 = module_0.Config(converters=none_type_0)
    none_type_1 = config_0.register_type(config_0)


def test_case_9():
    str_0 = ""
    config_0 = module_0.Config(str_0)
    none_type_0 = module_1.isclass(str_0)
    config_0.update_config(str_0)


def test_case_10():
    none_type_0 = None
    config_0 = module_0.Config(converters=none_type_0)
    str_0 = "7d\nh$qrbG~\t\x0b#"
    none_type_1 = config_0.__setitem__(str_0, config_0)
    set_0 = config_0.__getattr__(str_0)
    bytes_0 = b"Expectation Failed"
    bytes_1 = b'\x9c\xf2\xdd*\x8f\xac"\xf0\xd7"\t\xd1\xbbB'
    var_0 = module_1.isclass(set_0)
    tuple_0 = (set_0, bytes_0, bytes_1, var_0)
    config_0.__getattr__(tuple_0)


def test_case_11():
    none_type_0 = None
    config_0 = module_0.Config(converters=none_type_0)
    none_type_1 = config_0.register_type(none_type_0)
    config_0.register_type(none_type_0)


def test_case_12():
    none_type_0 = None
    config_0 = module_0.Config(converters=none_type_0)
    str_0 = "LOCAL_CERT_CREATOR"
    config_0.__setitem__(str_0, none_type_0)


def test_case_13():
    none_type_0 = None
    config_0 = module_0.Config(converters=none_type_0)
    str_0 = ""
    config_0.load_environment_vars(str_0)
