# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import codetiming._timer as module_0


def test_case_0():
    float_arg_0 = module_0.FloatArg()


def test_case_1():
    float_arg_0 = module_0.FloatArg()
    timer_error_0 = module_0.TimerError()
    timer_0 = module_0.Timer(initial_text=timer_error_0)
    none_type_0 = timer_0.start()
    timer_0.__enter__()


def test_case_2():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()


def test_case_3():
    timer_error_0 = module_0.TimerError()
    timer_0 = module_0.Timer(initial_text=timer_error_0)
    none_type_0 = timer_0.start()
    float_0 = timer_0.stop()
    timer_0.__exit__()


def test_case_4():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    none_type_0 = timer_1.__exit__()


def test_case_5():
    str_0 = "minutes"
    timer_0 = module_0.Timer(str_0)
    timer_error_0 = module_0.TimerError()
    timer_1 = timer_0.__enter__()
    var_0 = timer_0.__repr__()
    var_1 = timer_0.__repr__()
    none_type_0 = timer_0.__exit__()
    str_1 = "milliseconds"
    timer_error_1 = module_0.TimerError()
    timer_2 = module_0.Timer(str_1)
    var_2 = timer_1.__repr__()
    timer_1.stop()


def test_case_6():
    bool_0 = True
    none_type_0 = None
    timer_0 = module_0.Timer(logger=none_type_0)
    timer_1 = timer_0.__enter__()
    float_0 = timer_0.stop()
    timer_error_0 = module_0.TimerError()
    timer_2 = timer_0.__enter__()
    float_1 = timer_0.stop()
    module_0.FloatArg(**bool_0)


def test_case_7():
    timer_error_0 = module_0.TimerError()
    timer_0 = module_0.Timer(initial_text=timer_error_0)
    none_type_0 = timer_0.start()


def test_case_8():
    str_0 = "Dzbt}7i4mccs(J8"
    timer_0 = module_0.Timer(initial_text=str_0)
    timer_0.__enter__()


def test_case_9():
    float_arg_0 = module_0.FloatArg()
    timer_error_0 = module_0.TimerError()
    float_0 = -2527.0
    bool_0 = True
    dict_0 = {
        timer_error_0: float_0,
        timer_error_0: float_0,
        bool_0: bool_0,
        float_arg_0: bool_0,
    }
    timer_0 = module_0.Timer(float_arg_0, initial_text=float_arg_0, logger=dict_0)
    timer_0.start()
