# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import codetiming._timer as module_0
import collections as module_1


def test_case_0():
    timer_error_0 = module_0.TimerError()


def test_case_1():
    timer_0 = module_0.Timer()
    var_0 = timer_0.__repr__()
    timer_error_0 = module_0.TimerError()
    none_type_0 = timer_0.start()
    var_1 = timer_0.__repr__()
    timer_0.__enter__()


def test_case_2():
    float_arg_0 = module_0.FloatArg()
    timer_0 = module_0.Timer(initial_text=float_arg_0)
    none_type_0 = timer_0.start()


def test_case_3():
    none_type_0 = None
    timer_0 = module_0.Timer(initial_text=none_type_0)
    none_type_1 = timer_0.start()
    none_type_2 = timer_0.__exit__()
    timer_0.__exit__()


def test_case_4():
    float_arg_0 = module_0.FloatArg()
    timer_0 = module_0.Timer(initial_text=float_arg_0)
    none_type_0 = timer_0.start()
    none_type_1 = timer_0.__exit__()


def test_case_5():
    float_arg_0 = module_0.FloatArg()
    timer_0 = module_0.Timer(initial_text=float_arg_0)
    timer_1 = module_0.Timer(text=timer_0)
    timer_2 = timer_1.__enter__()
    float_0 = timer_1.stop()
    timer_3 = module_0.Timer(timer_2, initial_text=float_arg_0)


def test_case_6():
    none_type_0 = None
    timer_0 = module_0.Timer(none_type_0, none_type_0, logger=none_type_0)
    none_type_1 = timer_0.start()
    float_0 = timer_0.stop()
    var_0 = timer_0.__repr__()
    var_0.__enter__()


def test_case_7():
    none_type_0 = None
    timer_0 = module_0.Timer(initial_text=none_type_0)
    none_type_1 = timer_0.start()
    none_type_2 = timer_0.__exit__()


def test_case_8():
    float_arg_0 = module_0.FloatArg()
    float_arg_1 = module_0.FloatArg()
    list_0 = [float_arg_0, float_arg_0]
    timer_error_0 = module_0.TimerError(*list_0)
    timer_0 = module_0.Timer(float_arg_1, initial_text=float_arg_1)
    none_type_0 = timer_0.start()
    none_type_1 = timer_0.__exit__()
    user_dict_0 = module_1.UserDict()
    var_0 = user_dict_0.__iter__()
    timer_0.__exit__()


def test_case_9():
    str_0 = "W$.i5Yh\nBwV!#b"
    timer_0 = module_0.Timer(text=str_0, initial_text=str_0)
    timer_error_0 = module_0.TimerError()
    timer_1 = timer_0.__enter__()
    timer_0.__enter__()
