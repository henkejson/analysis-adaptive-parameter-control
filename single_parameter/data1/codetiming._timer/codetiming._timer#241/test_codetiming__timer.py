# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import codetiming._timer as module_0
import dataclasses as module_1
import codetiming._timers as module_2


def test_case_0():
    timer_error_0 = module_0.TimerError()


def test_case_1():
    timer_0 = module_0.Timer()
    none_type_0 = timer_0.start()
    none_type_1 = timer_0.__exit__()


def test_case_2():
    timer_error_0 = module_0.TimerError()
    timer_0 = module_0.Timer(timer_error_0)
    timer_0.__exit__()


def test_case_3():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    float_0 = timer_1.stop()


def test_case_4():
    none_type_0 = None
    timer_0 = module_0.Timer(none_type_0)
    timer_1 = timer_0.__enter__()
    timer_0.__enter__()


def test_case_5():
    complex_0 = -3435.4 - 1918.9304j
    timer_error_0 = module_0.TimerError()
    str_0 = "Yc"
    timer_0 = module_0.Timer(str_0)
    timer_1 = timer_0.__enter__()
    float_0 = timer_0.stop()
    complex_0.get(complex_0, complex_0)


def test_case_6():
    timer_error_0 = module_0.TimerError()
    dict_0 = {}
    none_type_0 = None
    timer_0 = module_0.Timer(logger=none_type_0)
    none_type_1 = timer_0.start()
    var_0 = module_1.field(default_factory=none_type_0, init=dict_0)
    var_0.__exit__()


def test_case_7():
    timer_error_0 = module_0.TimerError()
    timer_0 = module_0.Timer(initial_text=timer_error_0)
    none_type_0 = timer_0.start()
    none_type_1 = timer_0.__exit__()


def test_case_8():
    str_0 = "["
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0, str_0: str_0}
    var_0 = dict_0.__ror__(dict_0)
    timer_0 = module_0.Timer(str_0, initial_text=str_0, logger=dict_0)
    timers_0 = module_2.Timers()
    var_1 = timers_0.copy()
    var_2 = timers_0.update()
    var_3 = var_2.__eq__(timers_0)
    timer_0.start()


def test_case_9():
    timer_error_0 = module_0.TimerError()
    dict_0 = {}
    none_type_0 = None
    timer_0 = module_0.Timer(logger=none_type_0)
    none_type_1 = timer_0.start()
    list_0 = []
    none_type_2 = timer_0.__exit__(*list_0)
    var_0 = module_1.field(default_factory=none_type_0, init=dict_0)
    var_0.__exit__()


def test_case_10():
    timer_error_0 = module_0.TimerError()
    dict_0 = {timer_error_0: timer_error_0}
    timer_0 = module_0.Timer(timer_error_0, initial_text=timer_error_0)
    none_type_0 = timer_0.start()
    var_0 = module_1.field(repr=timer_0, compare=dict_0, kw_only=timer_0)
    none_type_1 = timer_0.__exit__()
    timer_0.stop()
