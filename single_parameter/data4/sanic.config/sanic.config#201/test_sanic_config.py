# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.config as module_0
import sanic.errorpages as module_1
import inspect as module_2


def test_case_0():
    config_0 = module_0.Config()


def test_case_1():
    bytes_0 = b"\x88-\x04\xd2\xe0\xedH!\x84\x89/\xd5\xb7"
    list_0 = [bytes_0]
    str_0 = 'Value "'
    dict_0 = {str_0: str_0}
    config_0 = module_0.Config(dict_0, converters=dict_0)
    str_0.register(list_0, list_0)


def test_case_2():
    bytes_0 = b":status"
    str_0 = "__slots__"
    str_1 = ",,9n1Q%IUNw7b"
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0, str_1: str_0}
    config_0 = module_0.Config(dict_0)
    none_type_0 = config_0.register_type(bytes_0)
    int_0 = 3066
    module_0.DescriptorMeta(int_0)


def test_case_3():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)
    none_type_0 = config_0.register_type(config_0)
    config_0.__getattr__(none_type_0)


def test_case_4():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)


def test_case_5():
    config_0 = module_0.Config()
    none_type_0 = config_0.register_type(config_0)
    config_0.update_config(none_type_0)


def test_case_6():
    bool_0 = True
    none_type_0 = None
    config_0 = module_0.Config(keep_alive=bool_0, converters=none_type_0)
    bool_1 = False
    module_1.check_error_format(bool_1)


def test_case_7():
    int_0 = -332
    dict_0 = {int_0: int_0, int_0: int_0}
    str_0 = "8bvhpXp#wvbY'SR"
    config_0 = module_0.Config(env_prefix=str_0)
    config_0.__setattr__(int_0, dict_0)


def test_case_8():
    config_0 = module_0.Config()
    str_0 = "HIq1"
    none_type_0 = config_0.__setitem__(str_0, str_0)
    var_0 = config_0.update_config(config_0)
    none_type_1 = config_0.register_type(config_0)
    config_0.update_config(none_type_1)


def test_case_9():
    config_0 = module_0.Config()
    str_0 = "middlewares"
    none_type_0 = config_0.__setattr__(str_0, config_0)
    config_0.update_config(str_0)


def test_case_10():
    config_0 = module_0.Config()
    config_0.__getattr__(config_0)


def test_case_11():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)
    config_0.update_config(var_0)


def test_case_12():
    config_0 = module_0.Config()
    none_type_0 = config_0.register_type(config_0)
    config_0.register_type(config_0)


def test_case_13():
    config_0 = module_0.Config()
    none_type_0 = None
    config_1 = module_0.Config(
        env_prefix=none_type_0, keep_alive=none_type_0, converters=none_type_0
    )
    var_0 = module_2.isclass(config_0)
    var_0.update_config(config_1)
