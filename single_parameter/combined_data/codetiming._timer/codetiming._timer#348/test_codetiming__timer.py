# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import codetiming._timer as module_0


def test_case_0():
    float_arg_0 = module_0.FloatArg()


def test_case_1():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    float_0 = timer_1.stop()


def test_case_2():
    timer_0 = module_0.Timer()
    str_0 = '"R\\f]V){\x0b'
    timer_1 = module_0.Timer(str_0, timer_0)
    timer_0.__exit__()


def test_case_3():
    bool_0 = True
    timer_0 = module_0.Timer(bool_0, initial_text=bool_0)
    none_type_0 = None
    none_type_1 = timer_0.start()
    var_0 = timer_0.__call__(none_type_0)
    float_0 = timer_0.stop()
    timer_1 = timer_0.__enter__()
    timer_0.__enter__()


def test_case_4():
    bool_0 = True
    timer_0 = module_0.Timer(bool_0, initial_text=bool_0)
    timer_1 = timer_0.__enter__()
    float_0 = timer_0.stop()


def test_case_5():
    str_0 = "6_mH"
    none_type_0 = None
    timer_0 = module_0.Timer(none_type_0, initial_text=str_0)
    var_0 = timer_0.__repr__()
    var_1 = timer_0.__eq__(var_0)
    timer_error_0 = module_0.TimerError()
    timer_1 = timer_0.__enter__()
    timer_1.start()


def test_case_6():
    bool_0 = True
    none_type_0 = None
    timer_0 = module_0.Timer(bool_0, logger=none_type_0)
    none_type_1 = timer_0.start()
    none_type_2 = timer_0.__exit__()
    timer_0.stop()


def test_case_7():
    bool_0 = True
    timer_0 = module_0.Timer(initial_text=bool_0)
    timer_1 = timer_0.__enter__()
    float_0 = timer_0.stop()


def test_case_8():
    bool_0 = False
    timer_0 = module_0.Timer(bool_0, initial_text=bool_0)
    timer_1 = module_0.Timer(text=timer_0)
    timer_2 = timer_1.__enter__()
    float_0 = timer_1.stop()
