# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import codetiming._timer as module_0
import codetiming._timers as module_1


def test_case_0():
    timer_error_0 = module_0.TimerError()


def test_case_1():
    none_type_0 = None
    timer_0 = module_0.Timer(none_type_0)
    timer_1 = timer_0.__enter__()


def test_case_2():
    bool_0 = True
    timer_0 = module_0.Timer(initial_text=bool_0)
    timer_1 = timer_0.__enter__()
    float_0 = timer_1.stop()
    timer_0.stop()


def test_case_3():
    bool_0 = False
    timer_0 = module_0.Timer(initial_text=bool_0)
    timer_1 = timer_0.__enter__()
    none_type_0 = timer_1.__exit__()


def test_case_4():
    bool_0 = True
    timer_0 = module_0.Timer(initial_text=bool_0)
    timer_1 = timer_0.__enter__()
    float_0 = timer_1.stop()


def test_case_5():
    float_arg_0 = module_0.FloatArg()
    timer_error_0 = module_0.TimerError()
    str_0 = "V~\x0c_;^9*gWNU:Fj"
    timer_0 = module_0.Timer(str_0, initial_text=str_0, logger=str_0)
    timer_0.start()


def test_case_6():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    float_0 = timer_1.stop()


def test_case_7():
    bool_0 = True
    timer_error_0 = module_0.TimerError()
    timer_0 = module_0.Timer(initial_text=bool_0)
    timer_1 = timer_0.__enter__()
    timer_0.__enter__()


def test_case_8():
    none_type_0 = None
    timer_0 = module_0.Timer(none_type_0, logger=none_type_0)
    list_0 = [timer_0]
    timer_error_0 = module_0.TimerError(*list_0)
    timer_1 = timer_0.__enter__()
    float_0 = timer_0.stop()


def test_case_9():
    bool_0 = True
    timer_0 = module_0.Timer(bool_0)
    timer_1 = timer_0.__enter__()
    float_0 = timer_0.stop()


def test_case_10():
    bool_0 = True
    timer_0 = module_0.Timer(bool_0, initial_text=bool_0)
    timer_1 = timer_0.__enter__()
    float_0 = timer_0.stop()


def test_case_11():
    bool_0 = True
    timer_0 = module_0.Timer(initial_text=bool_0)
    timer_1 = module_0.Timer(text=timer_0)
    timer_2 = timer_1.__enter__()
    float_0 = timer_2.stop()
    timers_0 = module_1.Timers()
    timers_0.__ior__(bool_0)
