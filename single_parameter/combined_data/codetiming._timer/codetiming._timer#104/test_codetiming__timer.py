# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import codetiming._timer as module_0


def test_case_0():
    float_arg_0 = module_0.FloatArg()


def test_case_1():
    timer_0 = module_0.Timer()
    timer_error_0 = module_0.TimerError()
    timer_1 = timer_0.__enter__()
    none_type_0 = timer_0.__exit__()
    timer_2 = timer_0.__enter__()
    float_arg_0 = module_0.FloatArg()
    timer_0.__enter__()


def test_case_2():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    float_0 = timer_1.stop()


def test_case_3():
    str_0 = "L-'\\2:k'JsV~8\x0cCq"
    none_type_0 = None
    timer_0 = module_0.Timer(none_type_0, str_0, str_0, none_type_0)
    timer_error_0 = module_0.TimerError()
    timer_1 = module_0.Timer()
    timer_1.stop()


def test_case_4():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    none_type_0 = timer_0.__exit__()
    float_arg_0 = module_0.FloatArg()
    var_0 = timer_0.__eq__(none_type_0)
    float_arg_1 = module_0.FloatArg()
    var_0.start()


def test_case_5():
    none_type_0 = None
    timer_error_0 = module_0.TimerError()
    dict_0 = {
        none_type_0: none_type_0,
        none_type_0: none_type_0,
        none_type_0: none_type_0,
        none_type_0: none_type_0,
    }
    str_0 = "T"
    timer_0 = module_0.Timer(initial_text=str_0, logger=none_type_0)
    timer_1 = timer_0.__enter__()
    timer_2 = module_0.Timer(text=dict_0)
    timer_3 = module_0.Timer(logger=str_0)
    var_0 = none_type_0.__repr__()
    none_type_1 = timer_0.__exit__(*var_0)
    var_0.items()


def test_case_6():
    float_arg_0 = module_0.FloatArg()
    float_arg_1 = module_0.FloatArg()
    timer_0 = module_0.Timer(initial_text=float_arg_0)
    none_type_0 = timer_0.start()


def test_case_7():
    str_0 = "qlq&~(r1/i#mFX:B$"
    timer_0 = module_0.Timer(str_0, initial_text=str_0, logger=str_0)
    timer_0.__enter__()


def test_case_8():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    none_type_0 = timer_0.__exit__()
    float_arg_0 = module_0.FloatArg()
    var_0 = timer_0.__eq__(none_type_0)
    var_1 = timer_1.__repr__()
    timer_2 = module_0.Timer(var_0, timer_0, float_arg_0)
    none_type_1 = timer_2.start()
    var_0.__enter__()


def test_case_9():
    str_0 = "_"
    none_type_0 = None
    timer_0 = module_0.Timer(str_0, none_type_0, logger=none_type_0)
    timer_1 = timer_0.__enter__()
    timer_2 = module_0.Timer()
    timer_3 = timer_2.__enter__()
    var_0 = timer_1.__eq__(timer_0)
    none_type_1 = timer_0.__exit__()
    none_type_2 = timer_2.__exit__()
    none_type_3 = timer_3.start()
    timer_3.start()
