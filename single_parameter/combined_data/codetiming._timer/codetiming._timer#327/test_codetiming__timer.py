# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import codetiming._timer as module_0


def test_case_0():
    float_arg_0 = module_0.FloatArg()


def test_case_1():
    bool_0 = True
    timer_0 = module_0.Timer(initial_text=bool_0)
    timer_1 = timer_0.__enter__()
    float_0 = timer_0.stop()
    timer_2 = timer_1.__enter__()
    timer_1.__enter__()


def test_case_2():
    bool_0 = True
    timer_0 = module_0.Timer(initial_text=bool_0)
    timer_1 = timer_0.__enter__()
    none_type_0 = timer_1.__exit__()


def test_case_3():
    str_0 = "minutes"
    timer_0 = module_0.Timer(text=str_0)
    timer_0.stop()


def test_case_4():
    none_type_0 = None
    timer_0 = module_0.Timer(initial_text=none_type_0)
    timer_error_0 = module_0.TimerError()
    none_type_1 = None
    timer_1 = module_0.Timer(initial_text=none_type_1, logger=none_type_1)
    none_type_2 = timer_1.start()
    var_0 = timer_1.__call__(timer_error_0)
    timer_1.start()


def test_case_5():
    bool_0 = False
    timer_0 = module_0.Timer(initial_text=bool_0)
    timer_1 = timer_0.__enter__()
    none_type_0 = timer_1.__exit__()


def test_case_6():
    bool_0 = True
    none_type_0 = None
    timer_0 = module_0.Timer(initial_text=none_type_0, logger=none_type_0)
    timer_1 = timer_0.__enter__()
    var_0 = timer_1.__call__(bool_0)
    var_1 = timer_0.__eq__(timer_0)
    var_2 = var_1.__eq__(timer_0)
    float_0 = timer_0.stop()
    float_arg_0 = module_0.FloatArg()
    none_type_1 = timer_0.start()
    var_0.__enter__()


def test_case_7():
    str_0 = "seconds"
    str_1 = "Timer"
    timer_0 = module_0.Timer(initial_text=str_1)
    timer_1 = module_0.Timer(str_0, initial_text=str_0)
    none_type_0 = timer_1.start()
    var_0 = timer_0.__repr__()
    var_0.clear()


def test_case_8():
    str_0 = "minutes"
    dict_0 = {str_0: str_0}
    timer_0 = module_0.Timer(str_0, dict_0, dict_0)
    timer_1 = timer_0.__enter__()
    module_0.FloatArg(**dict_0)


def test_case_9():
    bool_0 = True
    none_type_0 = None
    timer_0 = module_0.Timer(bool_0, initial_text=none_type_0)
    timer_1 = timer_0.__enter__()
    var_0 = timer_0.__repr__()
    var_1 = timer_0.__eq__(timer_0)
    float_0 = timer_0.stop()
    float_arg_0 = module_0.FloatArg()
    none_type_1 = timer_0.start()
    var_0.__enter__()
