# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import codetiming._timer as module_0
import dataclasses as module_1
import codetiming._timers as module_2


def test_case_0():
    timer_error_0 = module_0.TimerError()


def test_case_1():
    timer_0 = module_0.Timer()
    list_0 = timer_0.__call__(timer_0)
    timer_1 = timer_0.__enter__()
    timer_1.start()


def test_case_2():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()


def test_case_3():
    timer_0 = module_0.Timer()
    list_0 = []
    timer_1 = timer_0.__enter__()
    float_arg_0 = module_0.FloatArg(*list_0)
    none_type_0 = timer_1.__exit__()
    var_0 = module_1.dataclass(
        order=none_type_0,
        unsafe_hash=float_arg_0,
        frozen=none_type_0,
        match_args=timer_0,
        slots=none_type_0,
    )
    timer_0.__exit__()


def test_case_4():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    none_type_0 = timer_1.__exit__()


def test_case_5():
    str_0 = "2[W XZTb="
    timer_0 = module_0.Timer(str_0, initial_text=str_0)
    none_type_0 = timer_0.start()
    var_0 = timer_0.__repr__()


def test_case_6():
    timer_0 = module_0.Timer()
    list_0 = []
    timer_1 = timer_0.__call__(list_0)
    none_type_0 = None
    timer_2 = module_0.Timer(none_type_0, logger=none_type_0)
    timer_3 = timer_2.__enter__()
    var_0 = module_1.field(default_factory=none_type_0, init=timer_0, compare=timer_0)
    var_0.__delitem__(none_type_0)


def test_case_7():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    none_type_0 = timer_1.__exit__()
    timer_2 = module_0.Timer(timer_1)
    timer_3 = timer_2.__enter__()
    timer_3.__exit__()


def test_case_8():
    bool_0 = True
    set_0 = {bool_0, bool_0, bool_0}
    none_type_0 = None
    timer_0 = module_0.Timer(initial_text=set_0, logger=none_type_0)
    none_type_1 = timer_0.start()
    dict_0 = {}
    timers_0 = module_2.Timers(**dict_0)
    none_type_2 = timer_0.__exit__()
    timer_0.stop()


def test_case_9():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    timer_2 = module_0.Timer(timer_0, initial_text=timer_0)
    none_type_0 = timer_2.start()
    timer_2.stop()


def test_case_10():
    timer_0 = module_0.Timer()
    var_0 = timer_0.__repr__()
    var_1 = var_0.__repr__()
    timer_1 = module_0.Timer(initial_text=timer_0, logger=var_0)
    timer_2 = timer_0.__enter__()
    timer_3 = module_0.Timer(timer_1, initial_text=var_0)
    timer_1.start()


def test_case_11():
    timer_0 = module_0.Timer()
    var_0 = timer_0.__repr__()
    timer_1 = timer_0.__enter__()
    timer_2 = module_0.Timer(timer_0, initial_text=timer_0)
    timer_3 = module_0.Timer(var_0, timer_0)
    none_type_0 = timer_3.start()
    float_0 = timer_3.stop()
    var_0.__contains__(none_type_0)
