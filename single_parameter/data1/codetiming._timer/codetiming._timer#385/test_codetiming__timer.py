# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import codetiming._timer as module_0
import contextlib as module_1
import dataclasses as module_2


def test_case_0():
    timer_error_0 = module_0.TimerError()


def test_case_1():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()


def test_case_2():
    timer_0 = module_0.Timer()
    var_0 = timer_0.__repr__()
    var_1 = module_1.ContextDecorator()
    timer_0.stop()


def test_case_3():
    none_type_0 = None
    timer_0 = module_0.Timer(none_type_0)
    timer_1 = module_0.Timer(logger=none_type_0)
    list_0 = [none_type_0, none_type_0, none_type_0]
    timer_0.__exit__(*list_0)


def test_case_4():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    float_0 = timer_0.stop()


def test_case_5():
    timer_0 = module_0.Timer()
    none_type_0 = timer_0.start()
    timer_0.start()


def test_case_6():
    timer_0 = module_0.Timer()
    timer_1 = module_0.Timer(timer_0, timer_0, timer_0)
    none_type_0 = timer_1.start()
    timer_0.__exit__()


def test_case_7():
    none_type_0 = None
    timer_0 = module_0.Timer(none_type_0)
    timer_1 = module_0.Timer(logger=none_type_0)
    var_0 = timer_0.__call__(timer_0)
    none_type_1 = timer_1.start()


def test_case_8():
    timer_0 = module_0.Timer()
    var_0 = timer_0.__call__(timer_0)
    var_1 = timer_0.__call__(timer_0)
    none_type_0 = None
    timer_1 = timer_0.__enter__()
    none_type_1 = timer_0.__exit__()
    timer_2 = module_0.Timer(none_type_0, none_type_0, var_0)
    none_type_2 = timer_2.start()
    timer_3 = module_0.Timer(text=var_1, logger=var_1)
    timer_4 = module_0.Timer(none_type_0)
    timer_3.stop()


def test_case_9():
    timer_0 = module_0.Timer()
    var_0 = timer_0.__call__(timer_0)
    timer_1 = module_0.Timer(timer_0, var_0, timer_0)
    none_type_0 = timer_1.start()
    timer_2 = timer_0.__enter__()
    none_type_1 = timer_0.__exit__()
    timer_1.stop()


def test_case_10():
    timer_0 = module_0.Timer()
    var_0 = timer_0.__repr__()
    var_1 = timer_0.__call__(timer_0)
    none_type_0 = None
    timer_1 = timer_0.__enter__()
    none_type_1 = timer_0.__exit__()
    timer_2 = module_0.Timer(none_type_0, none_type_0, var_0)
    timer_2.start()


def test_case_11():
    timer_0 = module_0.Timer()
    list_0 = []
    timer_error_0 = module_0.TimerError(*list_0)
    var_0 = timer_0.__call__(timer_0)
    none_type_0 = None
    timer_1 = timer_0.__enter__()
    none_type_1 = timer_1.__exit__()
    timer_2 = module_0.Timer(logger=none_type_0)
    none_type_2 = timer_2.start()
    timer_3 = module_0.Timer()
    none_type_3 = timer_0.start()
    timer_4 = module_0.Timer(none_type_1, var_0)
    float_0 = timer_1.stop()
    var_1 = module_2.dataclass(
        frozen=none_type_1, match_args=none_type_0, slots=timer_4
    )
    none_type_4 = timer_2.__exit__()
    timer_2.stop()
