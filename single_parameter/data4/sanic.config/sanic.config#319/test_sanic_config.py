# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.config as module_0
import inspect as module_1


def test_case_0():
    config_0 = module_0.Config()


def test_case_1():
    config_0 = module_0.Config()
    str_0 = "n+8UBkp^1XQJ#+"
    bool_0 = False
    config_1 = module_0.Config(config_0, str_0, bool_0)


def test_case_2():
    str_0 = ",A)sm\x0c|\x0c.sp{iSS0 "
    dict_0 = {}
    config_0 = module_0.Config(dict_0)
    config_0.__getattr__(str_0)


def test_case_3():
    str_0 = ",A)sm\x0c|\x0c.sp{iSS0 "
    dict_0 = {}
    config_0 = module_0.Config(dict_0)
    none_type_0 = config_0.register_type(dict_0)
    config_1 = module_0.Config(keep_alive=config_0)
    config_0.update_config(str_0)


def test_case_4():
    dict_0 = {}
    config_0 = module_0.Config(dict_0)
    none_type_0 = config_0.register_type(dict_0)
    config_0.update_config(none_type_0)


def test_case_5():
    list_0 = []
    config_0 = module_0.Config()
    str_0 = "\\h;e:kqDE7_=hf"
    none_type_0 = None
    none_type_1 = config_0.__setitem__(str_0, none_type_0)
    none_type_2 = config_0.update(*list_0)
    config_1 = module_0.Config()
    str_1 = "Static file or directory must be a path-like object or string"
    none_type_3 = config_1.__setattr__(str_1, str_1)
    module_0.Config(env_prefix=config_1)


def test_case_6():
    str_0 = ",A)sm\x0c|\x0c.sp{iSS0 "
    bool_0 = True
    module_0.Config(keep_alive=bool_0, converters=str_0)


def test_case_7():
    config_0 = module_0.Config()
    config_0.__getattr__(config_0)


def test_case_8():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)


def test_case_9():
    config_0 = module_0.Config()
    none_type_0 = None
    config_0.update_config(none_type_0)


def test_case_10():
    none_type_0 = None
    var_0 = module_1.isdatadescriptor(none_type_0)
    var_1 = var_0.__bool__()
    none_type_1 = None
    config_0 = module_0.Config(none_type_1, none_type_1, converters=none_type_1)
    none_type_2 = config_0.register_type(var_1)


def test_case_11():
    dict_0 = {}
    config_0 = module_0.Config(dict_0)
    none_type_0 = None
    none_type_1 = config_0.register_type(dict_0)
    config_1 = module_0.Config(converters=config_0)
    config_1.__getattr__(none_type_0)


def test_case_12():
    dict_0 = {}
    config_0 = module_0.Config(dict_0, keep_alive=dict_0)
    list_0 = [config_0, dict_0, dict_0, config_0]
    none_type_0 = config_0.update(*list_0)
    str_0 = "LOCAL_CERT_CREATOR"
    config_0.__setattr__(str_0, dict_0)
