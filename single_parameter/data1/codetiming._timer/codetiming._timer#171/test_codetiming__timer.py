# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import codetiming._timer as module_0
import codetiming._timers as module_1


def test_case_0():
    float_arg_0 = module_0.FloatArg()


def test_case_1():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    float_0 = timer_1.stop()


def test_case_2():
    timers_0 = module_1.Timers()
    timer_0 = module_0.Timer(text=timers_0)
    timer_0.stop()


def test_case_3():
    none_type_0 = None
    str_0 = '"L[j.\\8%~Gz?~'
    timer_0 = module_0.Timer(none_type_0, str_0, str_0)
    timer_1 = timer_0.__enter__()
    var_0 = timer_1.__call__(str_0)
    float_0 = timer_1.stop()
    timer_0.__exit__()


def test_case_4():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    timer_1.__enter__()


def test_case_5():
    timers_0 = module_1.Timers()
    timer_0 = module_0.Timer(logger=timers_0)
    timer_1 = timer_0.__enter__()
    var_0 = timer_1.__repr__()
    timer_2 = module_0.Timer(initial_text=var_0)
    var_0.__enter__()


def test_case_6():
    timers_0 = module_1.Timers()
    timer_0 = module_0.Timer(text=timers_0)
    timer_1 = timer_0.__enter__()
    timer_2 = module_0.Timer(initial_text=timer_1)
    timer_3 = module_0.Timer(logger=timers_0)
    timer_4 = timer_3.__enter__()
    float_0 = timer_3.stop()


def test_case_7():
    timers_0 = module_1.Timers()
    timer_0 = module_0.Timer(text=timers_0)
    timer_1 = module_0.Timer(initial_text=timer_0)
    timer_2 = timer_1.__enter__()


def test_case_8():
    timers_0 = module_1.Timers()
    str_0 = "\rP%{_t ]KiJ_"
    timer_0 = module_0.Timer(text=str_0, initial_text=str_0)
    timer_0.__enter__()


def test_case_9():
    timers_0 = module_1.Timers()
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    var_0 = timer_1.__repr__()
    bool_0 = True
    timer_2 = module_0.Timer(var_0, initial_text=bool_0)
    timer_3 = timer_2.__enter__()
    float_0 = timer_2.stop()


def test_case_10():
    timers_0 = module_1.Timers()
    timer_0 = module_0.Timer(text=timers_0)
    timer_1 = timer_0.__enter__()
    var_0 = timers_0.clear()
    timer_2 = module_0.Timer(text=timer_1)
    timer_3 = timer_2.__enter__()
    float_0 = timer_2.stop()
