# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import codetiming._timer as module_0


def test_case_0():
    timer_error_0 = module_0.TimerError()


def test_case_1():
    timer_0 = module_0.Timer()
    none_type_0 = timer_0.start()
    float_0 = timer_0.stop()


def test_case_2():
    timer_0 = module_0.Timer()
    timer_0.stop()


def test_case_3():
    timer_0 = module_0.Timer()
    float_arg_0 = module_0.FloatArg()
    timer_1 = timer_0.__enter__()
    timer_2 = module_0.Timer(timer_0, initial_text=float_arg_0)


def test_case_4():
    timer_error_0 = module_0.TimerError()
    timer_0 = module_0.Timer(logger=timer_error_0)
    none_type_0 = None
    timer_1 = module_0.Timer(logger=none_type_0)
    list_0 = [none_type_0, timer_0, timer_error_0]
    timer_0.__exit__(*list_0)


def test_case_5():
    timer_error_0 = module_0.TimerError()
    timer_error_1 = module_0.TimerError()
    none_type_0 = None
    timer_0 = module_0.Timer(logger=none_type_0)
    timer_1 = timer_0.__enter__()
    list_0 = [none_type_0, timer_1, timer_error_0]
    none_type_1 = timer_1.__exit__(*list_0)


def test_case_6():
    timer_error_0 = module_0.TimerError()
    timer_0 = module_0.Timer(initial_text=timer_error_0)
    timer_1 = timer_0.__enter__()
    none_type_0 = timer_0.__exit__()
    var_0 = timer_0.__repr__()
    float_arg_0 = module_0.FloatArg()
    timer_2 = timer_1.__enter__()


def test_case_7():
    timer_0 = module_0.Timer()
    float_arg_0 = module_0.FloatArg()
    timer_1 = timer_0.__enter__()
    timer_2 = module_0.Timer(timer_0, initial_text=float_arg_0)
    timer_3 = timer_2.__enter__()
    timer_0.start()


def test_case_8():
    timer_0 = module_0.Timer()
    float_arg_0 = timer_0.__repr__()
    timer_1 = module_0.Timer(timer_0, initial_text=float_arg_0)
    timer_1.start()


def test_case_9():
    float_arg_0 = module_0.FloatArg()
    timer_0 = module_0.Timer(float_arg_0)
    none_type_0 = timer_0.start()
    var_0 = timer_0.__repr__()
    timer_1 = module_0.Timer(none_type_0)
    none_type_1 = timer_0.__exit__()
    var_1 = float_arg_0.__eq__(float_arg_0)
    var_1.__setitem__(var_1, var_1)
