# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import codetiming._timer as module_0
import builtins as module_1


def test_case_0():
    timer_error_0 = module_0.TimerError()


def test_case_1():
    timer_0 = module_0.Timer()
    none_type_0 = timer_0.start()
    timer_0.values()


def test_case_2():
    timer_0 = module_0.Timer()
    timer_0.stop()


def test_case_3():
    timer_0 = module_0.Timer()
    timer_1 = module_0.Timer()
    timer_2 = timer_1.__enter__()
    timer_error_0 = module_0.TimerError()
    timer_error_1 = module_0.TimerError()


def test_case_4():
    float_arg_0 = module_0.FloatArg()
    timer_0 = module_0.Timer(initial_text=float_arg_0)
    timer_1 = timer_0.__enter__()
    float_0 = timer_0.stop()
    timer_0.__exit__()


def test_case_5():
    list_0 = []
    timer_0 = module_0.Timer(initial_text=list_0, logger=list_0)
    none_type_0 = None
    timer_1 = timer_0.__enter__()
    timer_error_0 = timer_0.__call__(timer_0)
    float_0 = timer_1.stop()
    timer_2 = module_0.Timer(text=none_type_0)
    timer_3 = module_0.Timer(none_type_0)
    timer_error_1 = module_0.TimerError(*list_0)
    list_1 = []
    timer_error_2 = module_0.TimerError(*list_1)
    none_type_1 = timer_0.start()


def test_case_6():
    float_arg_0 = module_0.FloatArg()
    timer_0 = module_0.Timer(initial_text=float_arg_0)
    timer_1 = timer_0.__enter__()
    module_0.TimerError(**timer_0)


def test_case_7():
    str_0 = "Timer is not running. Use .start() to start it"
    timer_0 = module_0.Timer(initial_text=str_0)
    none_type_0 = timer_0.start()
    var_0 = timer_0.__call__(timer_0)
    var_1 = timer_0.__repr__()
    var_1.__getitem__(str_0)


def test_case_8():
    timer_0 = module_0.Timer()
    none_type_0 = timer_0.start()
    timer_0.start()


def test_case_9():
    timer_0 = module_0.Timer()
    none_type_0 = timer_0.start()
    float_0 = timer_0.stop()


def test_case_10():
    float_arg_0 = module_0.FloatArg()
    timer_0 = module_0.Timer(float_arg_0, initial_text=float_arg_0)
    timer_1 = timer_0.__enter__()
    int_0 = 2
    var_0 = timer_0.__call__(int_0)
    var_0.pop(var_0, int_0)


def test_case_11():
    float_arg_0 = module_0.FloatArg()
    none_type_0 = None
    str_0 = "D?6;%}`"
    timer_0 = module_0.Timer(str_0)
    timer_1 = timer_0.__enter__()
    var_0 = timer_0.__call__(none_type_0)
    none_type_1 = timer_0.__exit__()
    module_0.TimerError(*var_0)


def test_case_12():
    float_arg_0 = module_0.FloatArg()
    none_type_0 = None
    timer_0 = module_0.Timer(initial_text=float_arg_0)
    timer_1 = module_0.Timer(none_type_0, timer_0)
    timer_2 = timer_1.__enter__()
    var_0 = module_1.BaseException()
    none_type_1 = timer_1.__exit__()
    var_1 = var_0.__repr__()
    var_1.values()
