# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import codetiming._timer as module_0
import codetiming._timers as module_1


def test_case_0():
    timer_error_0 = module_0.TimerError()


def test_case_1():
    timer_0 = module_0.Timer()
    float_arg_0 = timer_0.__repr__()
    timer_1 = timer_0.__enter__()
    none_type_0 = timer_0.__exit__()
    float_arg_1 = module_0.FloatArg()


def test_case_2():
    timer_0 = module_0.Timer()
    float_arg_0 = module_0.FloatArg()
    timer_0.__exit__()


def test_case_3():
    bool_0 = True
    timer_0 = module_0.Timer(initial_text=bool_0)
    timer_1 = timer_0.__enter__()
    var_0 = timer_0.__eq__(timer_0)
    var_0.__iter__()


def test_case_4():
    bool_0 = True
    timer_0 = module_0.Timer(initial_text=bool_0)
    timer_1 = timer_0.__enter__()
    timer_1.__enter__()


def test_case_5():
    none_type_0 = None
    timer_0 = module_0.Timer(text=none_type_0, logger=none_type_0)
    bool_0 = True
    timer_1 = module_0.Timer(initial_text=bool_0)
    none_type_1 = timer_1.start()
    none_type_2 = timer_1.__exit__()
    timer_2 = timer_1.__enter__()
    var_0 = timer_1.__repr__()
    bool_1 = False
    timer_3 = timer_0.__enter__()
    var_1 = var_0.__repr__()
    timer_4 = module_0.Timer(bool_0, timer_1)
    var_2 = var_0.__repr__()
    var_2.__setitem__(bool_1, var_2)


def test_case_6():
    bool_0 = True
    timer_0 = module_0.Timer(bool_0, bool_0, bool_0)
    timer_1 = module_0.Timer(logger=timer_0)
    timer_2 = timer_0.__enter__()
    var_0 = timer_2.__call__(timer_2)
    var_1 = var_0.__repr__()
    var_0.__exit__()


def test_case_7():
    none_type_0 = None
    timer_0 = module_0.Timer(text=none_type_0, logger=none_type_0)
    none_type_1 = timer_0.start()
    none_type_2 = timer_0.__exit__()
    timer_1 = timer_0.__enter__()
    var_0 = timer_0.__repr__()
    var_1 = var_0.__eq__(var_0)
    timer_0.__enter__()


def test_case_8():
    bool_0 = True
    timer_0 = module_0.Timer(initial_text=bool_0)
    str_0 = "seconds"
    none_type_0 = None
    timer_1 = module_0.Timer(str_0, initial_text=bool_0, logger=none_type_0)
    none_type_1 = timer_1.start()
    none_type_2 = timer_1.__exit__()
    var_0 = timer_0.__repr__()
    var_1 = timer_0.__eq__(timer_1)
    timer_2 = timer_1.__enter__()
    var_2 = timer_1.__repr__()
    timer_3 = module_0.Timer(text=var_1)
    module_0.FloatArg(*var_0)


def test_case_9():
    none_type_0 = None
    timer_0 = module_0.Timer(logger=none_type_0)
    bool_0 = True
    timer_1 = module_0.Timer(text=timer_0, initial_text=bool_0)
    none_type_1 = timer_1.start()
    list_0 = [bool_0, bool_0, bool_0]
    none_type_2 = timer_1.__exit__(*list_0)
    var_0 = timer_0.__repr__()
    timer_2 = timer_1.__enter__()
    var_1 = timer_2.__repr__()
    timer_3 = module_0.Timer()
    float_arg_0 = module_0.FloatArg()
    module_0.TimerError(**var_1)


def test_case_10():
    str_0 = "\r*Ts-XB(GsuO432JHwZ"
    timer_0 = module_0.Timer(initial_text=str_0)
    str_1 = "+9SM5D"
    none_type_0 = timer_0.start()
    timers_0 = module_1.Timers()
    timers_0.__ior__(str_1)
