# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.config as module_0
import inspect as module_1
import sanic.utils as module_2


def test_case_0():
    config_0 = module_0.Config()


def test_case_1():
    config_0 = module_0.Config()
    none_type_0 = None
    config_1 = module_0.Config(config_0, none_type_0, none_type_0, converters=config_0)
    config_1.update_config(none_type_0)


def test_case_2():
    config_0 = module_0.Config()
    none_type_0 = None
    none_type_1 = None
    config_1 = module_0.Config(config_0, none_type_1, none_type_0, converters=config_0)
    var_0 = module_1.getmembers(none_type_0)
    var_1 = module_1.getmembers(config_1)
    var_1.__new__(none_type_0, none_type_1, var_0, none_type_0)


def test_case_3():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)


def test_case_4():
    none_type_0 = None
    config_0 = module_0.Config()
    config_0.update_config(none_type_0)


def test_case_5():
    config_0 = module_0.Config()
    none_type_0 = module_1.isdatadescriptor(config_0)
    none_type_1 = None
    none_type_2 = config_0.register_type(config_0)
    str_0 = "jw1Oo[\x0cH"
    none_type_3 = config_0.__setitem__(str_0, none_type_2)
    config_0.update_config(none_type_1)


def test_case_6():
    str_0 = "E]ByF.;A/i^"
    none_type_0 = None
    config_0 = module_0.Config(none_type_0, str_0)
    module_2.load_module_from_file_location(none_type_0, str_0)


def test_case_7():
    config_0 = module_0.Config()
    var_0 = config_0.update_config(config_0)
    none_type_0 = None
    none_type_1 = config_0.update(**config_0)
    config_1 = module_0.Config(config_0, none_type_0, config_0, converters=config_0)
    var_0.register(var_0, var_0)


def test_case_8():
    config_0 = module_0.Config()
    config_0.__getattr__(config_0)


def test_case_9():
    config_0 = module_0.Config()
    str_0 = "WBhC]y <0~\n#?\taX#"
    none_type_0 = config_0.__setattr__(str_0, str_0)
    none_type_1 = config_0.update()
    bytes_0 = b"%\x01g=s\xf0\x1f"
    config_0.update_config(bytes_0)


def test_case_10():
    config_0 = module_0.Config()
    none_type_0 = None
    none_type_1 = config_0.register_type(none_type_0)
    var_0 = config_0.update_config(config_0)
    str_0 = ";U:\\t%l"
    none_type_2 = module_1.isclass(none_type_0)
    none_type_3 = config_0.__setattr__(str_0, var_0)
    config_0.register_type(none_type_1)
