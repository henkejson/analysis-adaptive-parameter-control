# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import codetiming._timer as module_0
import dataclasses as module_1


def test_case_0():
    timer_0 = module_0.Timer()


def test_case_1():
    var_0 = module_1.dataclass()
    timer_0 = module_0.Timer(var_0, initial_text=var_0)
    none_type_0 = timer_0.start()
    timer_0.__enter__()


def test_case_2():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()


def test_case_3():
    none_type_0 = None
    timer_0 = module_0.Timer(logger=none_type_0)
    none_type_1 = timer_0.start()
    timer_error_0 = module_0.TimerError()
    float_0 = timer_0.stop()
    timer_0.stop()


def test_case_4():
    timer_0 = module_0.Timer()
    timer_0.stop()


def test_case_5():
    timer_0 = module_0.Timer()
    timer_0.__exit__()


def test_case_6():
    bool_0 = False
    timer_0 = module_0.Timer(text=bool_0)
    timer_1 = timer_0.__enter__()
    timer_0.__exit__()


def test_case_7():
    none_type_0 = None
    timer_0 = module_0.Timer(logger=none_type_0)
    none_type_1 = timer_0.__call__(none_type_0)
    timer_1 = module_0.Timer(text=timer_0)
    timer_2 = timer_1.__enter__()
    none_type_2 = timer_1.__exit__()
    timer_0.stop()


def test_case_8():
    bool_0 = True
    none_type_0 = None
    timer_0 = module_0.Timer(bool_0, bool_0, logger=none_type_0)
    none_type_1 = timer_0.start()
    float_0 = timer_0.stop()
    timer_1 = module_0.Timer(text=bool_0)
    timer_2 = timer_1.__enter__()
    timer_1.__exit__()


def test_case_9():
    bool_0 = True
    timer_0 = module_0.Timer(initial_text=bool_0, logger=bool_0)
    timer_0.__enter__()


def test_case_10():
    str_0 = "Z"
    timer_0 = module_0.Timer(initial_text=str_0)
    var_0 = timer_0.__repr__()
    var_1 = timer_0.__eq__(str_0)
    var_2 = timer_0.__eq__(timer_0)
    var_3 = timer_0.__eq__(var_2)
    none_type_0 = timer_0.start()
    var_3.clear()
