# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.config as module_0
import sanic.constants as module_1
import inspect as module_2
import sanic.helpers as module_3


def test_case_0():
    config_0 = module_0.Config()


def test_case_1():
    bool_0 = False
    config_0 = module_0.Config(bool_0)
    none_type_0 = config_0.register_type(bool_0)
    config_1 = module_0.Config(bool_0, keep_alive=bool_0)
    var_0 = module_1.LocalCertCreator.MKCERT
    var_1 = config_1.update_config(config_1)
    var_2 = module_0.Config(converters=var_0)
    var_2.__getattr__(var_2)


def test_case_2():
    bool_0 = False
    config_0 = module_0.Config(bool_0, keep_alive=bool_0)


def test_case_3():
    none_type_0 = None
    config_0 = module_0.Config(none_type_0, none_type_0)
    none_type_1 = None
    config_0.update_config(none_type_1)


def test_case_4():
    bool_0 = False
    config_0 = module_0.Config(bool_0, keep_alive=bool_0)
    var_0 = config_0.update_config(config_0)


def test_case_5():
    config_0 = module_0.Config()
    none_type_0 = config_0.update()
    var_0 = module_2.isdatadescriptor(none_type_0)
    str_0 = "J\tp;'B!7Eh"
    none_type_1 = config_0.__setattr__(str_0, config_0)
    module_0.Config(env_prefix=config_0, converters=none_type_1)


def test_case_6():
    bool_0 = False
    config_0 = module_0.Config(bool_0, keep_alive=bool_0)
    config_0.update_config(bool_0)


def test_case_7():
    str_0 = "SANIC_"
    none_type_0 = None
    bool_0 = True
    config_0 = module_0.Config()
    none_type_1 = config_0.register_type(bool_0)
    bytes_0 = b"\xae\x02T\xe5\xf1\x1by\xb4\xcbX\x89\xf0\xaf40\xbc4\xf8\xef"
    var_0 = module_2.getmembers(bytes_0)
    var_0.__setattr__(str_0, none_type_0)


def test_case_8():
    config_0 = module_0.Config()
    none_type_0 = config_0.update()
    none_type_1 = config_0.register_type(config_0)
    config_0.__setitem__(config_0, config_0)


def test_case_9():
    default_0 = module_3.Default()
    dict_0 = {default_0: default_0, default_0: default_0, default_0: default_0}
    none_type_0 = None
    str_0 = "sJ(f&zv2\\"
    config_0 = module_0.Config(env_prefix=str_0, keep_alive=dict_0)
    list_0 = [none_type_0, dict_0, config_0]
    bool_0 = True
    dict_1 = {none_type_0: config_0, str_0: list_0, default_0: config_0, bool_0: str_0}
    config_0.respond(dict_1)


def test_case_10():
    bool_0 = True
    module_0.Config(bool_0, keep_alive=bool_0)


def test_case_11():
    bool_0 = False
    config_0 = module_0.Config(bool_0, keep_alive=bool_0)
    str_0 = "Message body set in response on "
    config_0.update_config(str_0)


def test_case_12():
    config_0 = module_0.Config()
    config_0.__getattr__(config_0)


def test_case_13():
    bool_0 = False
    config_0 = module_0.Config(bool_0)
    none_type_0 = config_0.register_type(bool_0)
    config_1 = module_2.isclass(bool_0)
    config_0.register_type(bool_0)


def test_case_14():
    bool_0 = False
    config_0 = module_0.Config(bool_0, keep_alive=bool_0)
    str_0 = ""
    config_0.load_environment_vars(str_0)
