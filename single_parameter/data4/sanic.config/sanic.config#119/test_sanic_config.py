# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.config as module_0
import sanic.utils as module_1
import inspect as module_2


def test_case_0():
    config_0 = module_0.Config()


def test_case_1():
    str_0 = ""
    bool_0 = False
    set_0 = {str_0, bool_0, bool_0}
    config_0 = module_0.Config(converters=set_0)


def test_case_2():
    str_0 = ""
    bool_0 = False
    set_0 = {str_0, bool_0, bool_0}
    module_0.Config(env_prefix=set_0)


def test_case_3():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)
    module_1.load_module_from_file_location(config_0)


def test_case_4():
    str_0 = 'GLi\nuZju"G^pxaO-\r T$'
    none_type_0 = None
    config_0 = module_0.Config(none_type_0)
    config_0.update_config(str_0)


def test_case_5():
    str_0 = ""
    bool_0 = False
    config_0 = module_0.Config(env_prefix=str_0)
    config_0.load_environment_vars(bool_0)


def test_case_6():
    str_0 = ""
    config_0 = module_0.Config()
    config_0.load_environment_vars(str_0)


def test_case_7():
    str_0 = 'GLi\nuZju"G^pxaO-\r T$'
    module_0.Config(str_0)


def test_case_8():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)
    var_1 = module_2.isdatadescriptor(config_0)
    config_0.__getattr__(config_0)


def test_case_9():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)


def test_case_10():
    str_0 = ""
    config_0 = module_0.Config()
    none_type_0 = config_0.__setitem__(str_0, config_0)
    config_0.load_environment_vars(str_0)


def test_case_11():
    bool_0 = False
    config_0 = module_0.Config()
    var_0 = config_0.load_environment_vars()
    config_1 = module_0.Config(keep_alive=bool_0, converters=var_0)


def test_case_12():
    config_0 = module_0.Config()
    none_type_0 = None
    config_0.update_config(none_type_0)


def test_case_13():
    bool_0 = True
    config_0 = module_0.Config()
    none_type_0 = config_0.register_type(bool_0)
    config_0.register_type(bool_0)


def test_case_14():
    str_0 = "6\x0b\x0c{V4QN@p\rkn/laz"
    str_1 = "_FALLBACK_ERROR_FORMAT"
    bool_0 = False
    str_2 = "`^<fdEK`$"
    dict_0 = {str_0: str_0, str_1: bool_0, str_2: str_1}
    module_0.Config(dict_0, str_0)
