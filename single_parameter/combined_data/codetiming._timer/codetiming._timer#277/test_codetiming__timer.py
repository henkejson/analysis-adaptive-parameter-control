# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import codetiming._timer as module_0
import dataclasses as module_1


def test_case_0():
    timer_error_0 = module_0.TimerError()


def test_case_1():
    timer_error_0 = module_0.TimerError()
    str_0 = '\tPt="HjqiZQ'
    timer_0 = module_0.Timer(text=str_0, initial_text=timer_error_0)
    timer_1 = module_0.Timer()
    none_type_0 = timer_0.start()
    timer_0.__enter__()


def test_case_2():
    timer_0 = module_0.Timer()
    none_type_0 = timer_0.start()
    none_type_1 = timer_0.__exit__()


def test_case_3():
    timer_error_0 = module_0.TimerError()
    str_0 = '\tPt="HjqiZQ'
    timer_0 = module_0.Timer(text=str_0)
    timer_0.stop()


def test_case_4():
    timer_error_0 = module_0.TimerError()
    str_0 = "8q;E7^D"
    timer_0 = module_0.Timer(text=str_0, initial_text=timer_error_0)
    timer_1 = timer_0.__enter__()


def test_case_5():
    timer_error_0 = module_0.TimerError()
    str_0 = "~m{Z"
    timer_0 = module_0.Timer(text=str_0, initial_text=str_0)
    timer_0.start()


def test_case_6():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    float_0 = timer_1.stop()


def test_case_7():
    timer_error_0 = module_0.TimerError()
    none_type_0 = None
    timer_0 = module_0.Timer(timer_error_0, none_type_0, logger=none_type_0)
    none_type_1 = timer_0.start()
    float_0 = timer_0.stop()
    timer_1 = module_0.Timer()
    timer_2 = timer_0.__enter__()
    float_arg_0 = module_0.FloatArg()
    var_0 = timer_2.__repr__()
    var_1 = var_0.__len__()
    var_1.__len__()


def test_case_8():
    timer_error_0 = module_0.TimerError()
    str_0 = '\tPt="HjqiZQ'
    timer_0 = module_0.Timer(str_0, timer_error_0, timer_error_0)
    none_type_0 = timer_0.start()
    timer_1 = module_0.Timer(text=str_0)
    none_type_1 = timer_1.start()
    float_0 = timer_1.stop()
    none_type_2 = timer_1.start()
    timer_2 = module_0.Timer(text=str_0, initial_text=timer_error_0)
    var_0 = module_1.dataclass(unsafe_hash=none_type_1, kw_only=str_0)
    timer_error_1 = module_0.TimerError()
    var_1 = var_0.__repr__()
    var_1.items()
