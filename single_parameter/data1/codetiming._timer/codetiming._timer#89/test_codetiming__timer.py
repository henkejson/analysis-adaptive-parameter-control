# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import codetiming._timer as module_0
import codetiming._timers as module_1


def test_case_0():
    timer_error_0 = module_0.TimerError()


def test_case_1():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    var_0 = timer_0.__eq__(timer_0)
    timer_1.start()


def test_case_2():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    float_arg_0 = module_0.FloatArg()


def test_case_3():
    str_0 = "LQvHGS~"
    int_0 = 1247
    timer_0 = module_0.Timer(initial_text=int_0)
    timer_1 = module_0.Timer(text=str_0, initial_text=str_0)
    timer_1.__exit__()


def test_case_4():
    timer_0 = module_0.Timer()
    none_type_0 = timer_0.start()
    none_type_1 = timer_0.__exit__()


def test_case_5():
    timer_error_0 = module_0.TimerError()
    timer_0 = module_0.Timer(text=timer_error_0, initial_text=timer_error_0)
    float_arg_0 = module_0.FloatArg()
    none_type_0 = timer_0.start()
    timer_error_1 = module_0.TimerError()
    module_0.FloatArg(*timer_0)


def test_case_6():
    timer_error_0 = module_0.TimerError()
    str_0 = "R8|$k}sng&lS*Rp"
    timer_0 = module_0.Timer(text=timer_error_0, initial_text=str_0)
    timer_1 = module_0.Timer()
    var_0 = timer_1.__repr__()
    float_arg_0 = module_0.FloatArg()
    timer_error_1 = module_0.TimerError()
    timer_0.start()


def test_case_7():
    float_arg_0 = module_0.FloatArg()
    str_0 = "Timer is not running. Use .start() to start it"
    none_type_0 = None
    timer_0 = module_0.Timer(none_type_0, str_0, logger=none_type_0)
    float_arg_1 = module_0.FloatArg()
    bool_0 = True
    none_type_1 = timer_0.start()
    list_0 = [float_arg_0, timer_0, bool_0, str_0]
    none_type_2 = timer_0.__exit__(*list_0)
    var_0 = timer_0.__eq__(timer_0)
    timer_1 = module_0.Timer()
    timers_0 = module_1.Timers()
    var_1 = timers_0.__ror__(timer_1)
    var_1.__ror__(str_0)


def test_case_8():
    timer_error_0 = module_0.TimerError()
    none_type_0 = None
    timer_0 = module_0.Timer()
    none_type_1 = timer_0.start()
    dict_0 = {}
    timer_error_1 = module_0.TimerError()
    var_0 = timer_0.__eq__(none_type_1)
    var_1 = var_0.__repr__()
    timer_1 = module_0.Timer(var_0, dict_0, var_0)
    timer_2 = timer_1.__enter__()
    var_0.__setitem__(none_type_0, var_1)


def test_case_9():
    float_arg_0 = module_0.FloatArg()
    str_0 = "Timer is not running. Use .start() to start it"
    timer_0 = module_0.Timer(str_0)
    float_arg_1 = module_0.FloatArg()
    bool_0 = True
    none_type_0 = timer_0.start()
    list_0 = [float_arg_0, timer_0, bool_0, str_0]
    none_type_1 = timer_0.__exit__(*list_0)
    var_0 = timer_0.__eq__(timer_0)
    timer_1 = module_0.Timer()
    timers_0 = module_1.Timers()
    var_1 = timers_0.__ror__(timer_1)
    var_1.__ror__(str_0)
