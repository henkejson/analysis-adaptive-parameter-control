# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import codetiming._timer as module_0
import dataclasses as module_1


def test_case_0():
    float_arg_0 = module_0.FloatArg()


def test_case_1():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    timer_0.__enter__()


def test_case_2():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    float_0 = timer_1.stop()


def test_case_3():
    timer_0 = module_0.Timer()
    timer_0.stop()


def test_case_4():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    var_0 = timer_1.__repr__()
    var_1 = timer_0.__repr__()
    none_type_0 = timer_1.__exit__(*var_0)
    timer_2 = module_0.Timer()


def test_case_5():
    timer_error_0 = module_0.TimerError()
    list_0 = []
    timer_0 = module_0.Timer(logger=list_0)
    var_0 = timer_0.__repr__()
    none_type_0 = timer_0.start()
    var_1 = module_1.field(compare=list_0)
    var_1.start()


def test_case_6():
    none_type_0 = None
    none_type_1 = None
    none_type_2 = None
    timer_0 = module_0.Timer(
        text=none_type_1, initial_text=none_type_2, logger=none_type_0
    )
    none_type_3 = timer_0.start()
    none_type_4 = timer_0.__exit__()
    var_0 = timer_0.__eq__(none_type_0)
    var_1 = timer_0.__call__(timer_0)
    var_0.__contains__(none_type_0)


def test_case_7():
    bool_0 = True
    timer_0 = module_0.Timer(initial_text=bool_0)
    none_type_0 = timer_0.start()
    timer_error_0 = module_0.TimerError()
    float_0 = timer_0.stop()
    timer_error_1 = module_0.TimerError()
    timer_0.__exit__()


def test_case_8():
    str_0 = "nB09/X\x0bPLe"
    timer_0 = module_0.Timer(initial_text=str_0, logger=str_0)
    timer_1 = module_0.Timer()
    timer_0.__enter__()


def test_case_9():
    timer_0 = module_0.Timer()
    none_type_0 = None
    var_0 = timer_0.__call__(none_type_0)
    timer_1 = timer_0.__enter__()
    timer_2 = module_0.Timer(timer_1, timer_1, timer_1)
    timer_3 = timer_2.__enter__()
    var_1 = timer_1.__repr__()
    none_type_1 = timer_0.__exit__()
    timer_4 = module_0.Timer(var_1)
    timer_3.stop()
