# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import codetiming._timer as module_0
import codetiming._timers as module_1


def test_case_0():
    timer_error_0 = module_0.TimerError()


def test_case_1():
    bool_0 = True
    timer_0 = module_0.Timer(initial_text=bool_0)
    timer_1 = timer_0.__enter__()
    timer_1.__enter__()


def test_case_2():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()


def test_case_3():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    float_0 = timer_1.stop()
    timer_0.stop()


def test_case_4():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    float_0 = timer_1.stop()


def test_case_5():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__call__(timer_0)
    float_0 = timer_1.__call__(timer_1)


def test_case_6():
    none_type_0 = None
    timer_error_0 = module_0.TimerError()
    list_0 = [none_type_0, none_type_0]
    str_0 = "E4 <a?jxk#"
    timer_0 = module_0.Timer(text=str_0, logger=none_type_0)
    timer_error_1 = module_0.TimerError()
    timer_1 = module_0.Timer(
        text=timer_error_0, initial_text=list_0, logger=none_type_0
    )
    none_type_1 = timer_0.start()
    list_0.__enter__()


def test_case_7():
    bool_0 = True
    timer_0 = module_0.Timer(initial_text=bool_0)
    timer_1 = timer_0.__enter__()


def test_case_8():
    str_0 = "`\x0csf\x0bNh\rp_l!XinHg"
    timer_0 = module_0.Timer(str_0, initial_text=str_0)
    timer_1 = timer_0.__enter__()
    none_type_0 = None
    dict_0 = {none_type_0: none_type_0}
    module_1.Timers(**dict_0)


def test_case_9():
    bool_0 = True
    timer_0 = module_0.Timer(bool_0)
    var_0 = timer_0.__call__(timer_0)
    bool_1 = False
    var_1 = var_0.__call__(bool_1)


def test_case_10():
    timer_0 = module_0.Timer()
    str_0 = "Timer stared"
    none_type_0 = None
    timer_1 = module_0.Timer(text=timer_0, initial_text=str_0, logger=none_type_0)
    none_type_1 = timer_1.start()
    none_type_2 = timer_1.__exit__()
    timer_0.stop()


def test_case_11():
    bool_0 = True
    str_0 = ")1xFupI}l<MSR;w@s?"
    timer_0 = module_0.Timer(str_0, initial_text=bool_0, logger=bool_0)
    timer_0.start()


def test_case_12():
    bool_0 = True
    timer_0 = module_0.Timer(initial_text=bool_0)
    none_type_0 = None
    bool_1 = True
    timer_1 = module_0.Timer(none_type_0, timer_0, bool_1)
    var_0 = timer_1.__call__(timer_0)
    var_1 = var_0.__call__(timer_1)
