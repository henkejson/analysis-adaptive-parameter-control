# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import codetiming._timer as module_0


def test_case_0():
    timer_error_0 = module_0.TimerError()


def test_case_1():
    timer_0 = module_0.Timer()
    none_type_0 = timer_0.start()
    timer_0.__enter__()


def test_case_2():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    float_0 = timer_0.stop()


def test_case_3():
    float_arg_0 = module_0.FloatArg()
    float_0 = 1125.625
    timer_0 = module_0.Timer(initial_text=float_0)
    none_type_0 = timer_0.start()
    none_type_1 = timer_0.__exit__()
    timer_0.__exit__()


def test_case_4():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    none_type_0 = timer_0.__exit__()


def test_case_5():
    float_0 = 1125.625
    timer_0 = module_0.Timer(initial_text=float_0)
    none_type_0 = timer_0.start()
    none_type_1 = timer_0.__exit__()


def test_case_6():
    float_0 = 1125.625
    timer_0 = module_0.Timer(float_0)
    none_type_0 = timer_0.start()
    none_type_1 = timer_0.__exit__()


def test_case_7():
    timer_0 = module_0.Timer()
    str_0 = ">cVj2A<K=KNGV\tM7;#ec"
    timer_1 = timer_0.__enter__()
    timer_2 = module_0.Timer(initial_text=str_0)
    none_type_0 = timer_2.start()
    none_type_1 = timer_2.__exit__()
    timer_3 = timer_1.__repr__()


def test_case_8():
    float_arg_0 = module_0.FloatArg()
    float_0 = 1125.625
    timer_0 = module_0.Timer(initial_text=float_0)
    none_type_0 = timer_0.start()
    none_type_1 = timer_0.__exit__()
    timer_1 = module_0.Timer(logger=none_type_0)
    timer_2 = timer_1.__enter__()


def test_case_9():
    int_0 = -2370
    float_0 = 1125.625
    none_type_0 = None
    timer_0 = module_0.Timer(int_0, logger=none_type_0)
    timer_1 = module_0.Timer(text=float_0, logger=none_type_0)
    var_0 = timer_1.__eq__(none_type_0)
    none_type_1 = timer_1.start()
    none_type_2 = timer_1.__exit__()
    timer_2 = timer_1.__enter__()
    var_0.__setitem__(var_0, timer_2)


def test_case_10():
    float_arg_0 = module_0.FloatArg()
    timer_0 = module_0.Timer(float_arg_0, initial_text=float_arg_0)
    none_type_0 = timer_0.start()
    none_type_1 = timer_0.__exit__()
    timer_1 = timer_0.__enter__()


def test_case_11():
    float_0 = 1125.1129041757029
    timer_0 = module_0.Timer(initial_text=float_0)
    timer_1 = module_0.Timer(text=timer_0)
    none_type_0 = timer_1.start()
    none_type_1 = timer_1.__exit__()
    timer_2 = timer_1.__enter__()
