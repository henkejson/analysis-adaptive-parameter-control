# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import codetiming._timer as module_0


def test_case_0():
    float_arg_0 = module_0.FloatArg()


def test_case_1():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    float_0 = timer_0.stop()


def test_case_2():
    float_0 = -3290.7
    timer_0 = module_0.Timer(logger=float_0)
    none_type_0 = None
    var_0 = timer_0.__eq__(none_type_0)
    timer_0.__exit__()


def test_case_3():
    timer_error_0 = module_0.TimerError()
    timer_0 = module_0.Timer(initial_text=timer_error_0)
    timer_error_1 = module_0.TimerError()
    timer_1 = timer_0.__enter__()
    timer_0.__enter__()


def test_case_4():
    str_0 = "v<xfFRi\\39\ty|5 f3Gv"
    dict_0 = {str_0: str_0}
    timer_0 = module_0.Timer(text=str_0, initial_text=str_0)
    none_type_0 = timer_0.start()
    float_arg_0 = module_0.FloatArg()
    none_type_1 = timer_0.__exit__()
    timer_1 = module_0.Timer(initial_text=dict_0)
    var_0 = timer_1.__repr__()
    var_1 = timer_1.__eq__(str_0)
    module_0.FloatArg(**dict_0)


def test_case_5():
    timer_error_0 = module_0.TimerError()
    none_type_0 = None
    timer_0 = module_0.Timer(initial_text=none_type_0, logger=none_type_0)
    timer_error_1 = module_0.TimerError()
    timer_error_2 = module_0.TimerError()
    timer_1 = timer_0.__enter__()
    float_0 = timer_1.stop()
    none_type_1 = timer_1.start()
    var_0 = timer_0.__repr__()
    timer_1.__enter__()


def test_case_6():
    timer_error_0 = module_0.TimerError()
    timer_error_1 = module_0.TimerError()
    timer_0 = module_0.Timer(
        timer_error_0, initial_text=timer_error_1, logger=timer_error_0
    )
    timer_error_2 = module_0.TimerError()
    timer_error_3 = module_0.TimerError()
    timer_0.__enter__()


def test_case_7():
    timer_error_0 = module_0.TimerError()
    none_type_0 = None
    timer_0 = module_0.Timer(timer_error_0, initial_text=none_type_0)
    timer_error_1 = module_0.TimerError()
    timer_1 = timer_0.__enter__()
    float_0 = timer_1.stop()
    none_type_1 = timer_1.start()
    timer_0.__enter__()
