# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import codetiming._timer as module_0
import codetiming._timers as module_1


def test_case_0():
    timer_error_0 = module_0.TimerError()


def test_case_1():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    none_type_0 = timer_0.__exit__()


def test_case_2():
    timer_0 = module_0.Timer()
    timer_0.__exit__()


def test_case_3():
    bool_0 = True
    timer_0 = module_0.Timer(bool_0, initial_text=bool_0)
    none_type_0 = timer_0.start()
    timer_error_0 = module_0.TimerError()
    none_type_1 = timer_0.__exit__()


def test_case_4():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    timers_0 = module_1.Timers()
    var_0 = timers_0.__len__()
    timer_0.__enter__()


def test_case_5():
    none_type_0 = None
    float_arg_0 = module_0.FloatArg()
    timer_0 = module_0.Timer(logger=none_type_0)
    none_type_1 = timer_0.start()
    str_0 = "Timer"
    dict_0 = {str_0: float_arg_0, str_0: none_type_1}
    timer_error_0 = timer_0.__repr__()
    none_type_2 = timer_0.__exit__()
    module_0.TimerError(**dict_0)


def test_case_6():
    float_arg_0 = module_0.FloatArg()
    bool_0 = True
    timer_0 = module_0.Timer(text=float_arg_0, initial_text=bool_0)
    timer_1 = timer_0.__enter__()
    timer_0.__exit__()


def test_case_7():
    none_type_0 = None
    float_arg_0 = module_0.FloatArg()
    str_0 = "f~+>kA25"
    timer_0 = module_0.Timer(none_type_0, none_type_0, str_0)
    none_type_1 = timer_0.start()
    timer_error_0 = module_0.TimerError()
    timer_0.__exit__()


def test_case_8():
    none_type_0 = None
    float_arg_0 = module_0.FloatArg()
    list_0 = [none_type_0, none_type_0, none_type_0, none_type_0]
    bool_0 = True
    timer_0 = module_0.Timer(bool_0, initial_text=bool_0)
    timer_1 = module_0.Timer(list_0, timer_0, logger=timer_0)
    none_type_1 = timer_1.start()
    timer_error_0 = module_0.TimerError()
    timer_1.__exit__(*list_0)
