# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import codetiming._timer as module_0
import codetiming._timers as module_1
import dataclasses as module_2


def test_case_0():
    timer_error_0 = module_0.TimerError()


def test_case_1():
    timer_0 = module_0.Timer()
    timer_1 = module_0.Timer(initial_text=timer_0)
    timer_2 = timer_1.__enter__()
    var_0 = timer_2.__call__(timer_0)
    timer_1.start()


def test_case_2():
    str_0 = "3<"
    timer_0 = module_0.Timer(str_0)
    timer_error_0 = timer_0.__call__(str_0)
    none_type_0 = timer_0.start()


def test_case_3():
    timer_0 = module_0.Timer()
    float_arg_0 = module_0.FloatArg()
    timer_0.stop()


def test_case_4():
    str_0 = "3<"
    timer_0 = module_0.Timer(str_0)
    var_0 = timer_0.__call__(str_0)
    var_0.__call__(timer_0)


def test_case_5():
    bool_0 = False
    none_type_0 = None
    timer_error_0 = module_0.TimerError()
    list_0 = [bool_0, bool_0, bool_0, bool_0]
    timer_0 = module_0.Timer(list_0, none_type_0, list_0, none_type_0)
    timer_1 = timer_0.__enter__()
    module_1.Timers(*list_0)


def test_case_6():
    timer_0 = module_0.Timer()
    none_type_0 = timer_0.start()
    float_0 = timer_0.stop()
    var_0 = module_2.field()
    float_arg_0 = module_0.FloatArg()


def test_case_7():
    str_0 = ":NDxIYMe0\tg"
    timer_0 = module_0.Timer(str_0, initial_text=str_0)
    timer_1 = timer_0.__enter__()
    var_0 = timer_1.__repr__()
    var_0.stop()


def test_case_8():
    bool_0 = False
    none_type_0 = None
    timer_error_0 = module_0.TimerError()
    list_0 = [bool_0, bool_0, bool_0, bool_0]
    timer_0 = module_0.Timer(list_0, none_type_0, list_0, none_type_0)
    timer_1 = timer_0.__enter__()
    timer_1.__exit__()


def test_case_9():
    timer_0 = module_0.Timer()
    none_type_0 = timer_0.start()
    float_0 = timer_0.stop()
    str_0 = "1FW:^zFbwu[sk/W<+"
    timer_1 = module_0.Timer()
    float_arg_0 = module_0.FloatArg()
    var_0 = timer_0.__eq__(str_0)
    timer_2 = module_0.Timer(float_arg_0, timer_1, var_0)
    timer_3 = timer_2.__enter__()
    var_1 = timer_3.__repr__()
    float_1 = timer_3.stop()
    var_0.__setitem__(timer_3, float_arg_0)
