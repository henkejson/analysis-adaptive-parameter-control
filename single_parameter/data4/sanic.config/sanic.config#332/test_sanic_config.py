# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.config as module_0
import inspect as module_1


def test_case_0():
    config_0 = module_0.Config()


def test_case_1():
    int_0 = 281
    none_type_0 = None
    module_0.Config(keep_alive=none_type_0, converters=int_0)


def test_case_2():
    none_type_0 = None
    config_0 = module_0.Config(env_prefix=none_type_0)
    config_0.update_config(none_type_0)


def test_case_3():
    none_type_0 = None
    config_0 = module_0.Config(env_prefix=none_type_0)
    list_0 = [config_0, config_0]
    config_1 = module_0.Config(config_0)
    none_type_1 = config_0.update(*list_0)
    none_type_2 = None
    config_0.update_config(none_type_2)


def test_case_4():
    config_0 = module_0.Config()
    none_type_0 = None
    config_0.update_config(none_type_0)


def test_case_5():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)


def test_case_6():
    str_0 = "7HK:;H+.mAp6t`S"
    var_0 = module_1.isclass(str_0)
    module_0.Config(converters=str_0)


def test_case_7():
    int_0 = 1937
    dict_0 = {}
    none_type_0 = None
    module_0.Config(dict_0, int_0, none_type_0)


def test_case_8():
    int_0 = 253
    config_0 = module_0.Config(keep_alive=int_0)
    none_type_0 = None
    config_0.update_config(none_type_0)


def test_case_9():
    none_type_0 = None
    config_0 = module_0.Config(env_prefix=none_type_0)
    var_0 = module_1.getmembers(config_0)
    list_0 = [var_0, var_0]
    none_type_1 = config_0.update(*list_0)
    config_0.update(*config_0)


def test_case_10():
    none_type_0 = None
    config_0 = module_0.Config(env_prefix=none_type_0)
    var_0 = module_1.getmembers(config_0)
    list_0 = [var_0, var_0]
    none_type_1 = config_0.update(*list_0)
    none_type_2 = config_0.update(*list_0)
    config_0.__getattr__(var_0)


def test_case_11():
    config_0 = module_0.Config()
    config_0.__setitem__(config_0, config_0)


def test_case_12():
    int_0 = 4096
    var_0 = module_1.isclass(int_0)
    none_type_0 = None
    str_0 = var_0.__str__()
    config_0 = module_0.Config(converters=none_type_0)
    var_1 = module_0.Config(converters=str_0)
    str_1 = "D6bn|ik(W"
    str_2 = "multiprocessing"
    dict_0 = {str_1: int_0, str_1: var_0, str_1: var_0, str_2: var_0}
    var_1.__subclasscheck__(str_2, dict_0)


def test_case_13():
    int_0 = 253
    config_0 = module_0.Config(keep_alive=int_0)
    none_type_0 = config_0.update(**config_0)
    none_type_1 = config_0.update()
    bytes_0 = b"\x816\x7f\x06"
    config_0.update_config(bytes_0)


def test_case_14():
    none_type_0 = None
    config_0 = module_0.Config(env_prefix=none_type_0)
    var_0 = module_1.getmembers(config_0)
    list_0 = [var_0, var_0]
    config_1 = module_0.Config(keep_alive=var_0)
    none_type_1 = config_0.update(*list_0)
    config_2 = module_0.Config(config_0)
    var_1 = config_2.update_config(config_0)
    none_type_2 = config_0.update(*list_0)
    module_0.DescriptorMeta(var_0)
