# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import codetiming._timer as module_0
import codetiming._timers as module_1


def test_case_0():
    timer_0 = module_0.Timer()


def test_case_1():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()


def test_case_2():
    none_type_0 = None
    timer_0 = module_0.Timer(none_type_0, initial_text=none_type_0)
    timer_0.stop()


def test_case_3():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    none_type_0 = timer_1.__exit__()


def test_case_4():
    timer_error_0 = module_0.TimerError()
    none_type_0 = None
    timer_0 = module_0.Timer(logger=none_type_0)
    timers_0 = module_1.Timers()
    list_0 = [none_type_0]
    timer_error_1 = module_0.TimerError(*list_0)
    timer_1 = timer_0.__enter__()
    float_0 = timer_0.stop()
    timer_2 = timer_0.__enter__()
    float_1 = timer_1.stop()
    none_type_1 = None
    timer_3 = module_0.Timer(none_type_1, list_0, logger=none_type_1)
    timer_4 = module_0.Timer(float_0)


def test_case_5():
    str_0 = "`GFMIM;5~5Od~-?l"
    timer_0 = module_0.Timer(initial_text=str_0)
    timer_1 = timer_0.__enter__()
    timer_0.__enter__()


def test_case_6():
    none_type_0 = None
    timer_0 = module_0.Timer(none_type_0, initial_text=none_type_0)
    timer_1 = module_0.Timer(initial_text=timer_0)
    timer_2 = timer_1.__enter__()


def test_case_7():
    str_0 = "L5L"
    timer_0 = module_0.Timer(initial_text=str_0)
    timer_1 = timer_0.__enter__()
    list_0 = timer_0.__call__(str_0)
    timer_1.__exit__(*list_0)


def test_case_8():
    str_0 = "9X3a9ZCJk/:VX=&tllR!"
    list_0 = [str_0]
    timer_0 = module_0.Timer(str_0, initial_text=list_0)
    timer_1 = timer_0.__enter__()
    none_type_0 = timer_1.__exit__(*list_0)
    none_type_1 = timer_1.start()
    dict_0 = {}
    timer_2 = module_0.Timer(none_type_0)
    timers_0 = module_1.Timers(**dict_0)
    none_type_2 = timer_2.start()
    timer_1.__exit__(*none_type_2)
