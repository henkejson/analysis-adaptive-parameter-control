# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import codetiming._timer as module_0
import dataclasses as module_1
import collections as module_2
import codetiming._timers as module_3


def test_case_0():
    timer_error_0 = module_0.TimerError()


def test_case_1():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    timer_0.__enter__()


def test_case_2():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()


def test_case_3():
    timer_0 = module_0.Timer()
    timer_0.stop()


def test_case_4():
    timer_0 = module_0.Timer()
    none_type_0 = timer_0.start()
    float_0 = timer_0.stop()


def test_case_5():
    float_arg_0 = module_0.FloatArg()
    float_arg_1 = module_0.FloatArg()
    timer_0 = module_0.Timer(initial_text=float_arg_1)
    timer_1 = timer_0.__enter__()
    timer_2 = module_0.Timer()


def test_case_6():
    timer_0 = module_0.Timer()
    none_type_0 = timer_0.start()
    none_type_1 = timer_0.__exit__()


def test_case_7():
    str_0 = "m^bD"
    timer_0 = module_0.Timer(initial_text=str_0)
    none_type_0 = timer_0.start()
    bytes_0 = b"24\x8dXx\xc5\xf1\xe1Y\xed\xcb\xca\xd7%"
    str_1 = "I]k\\oC/ZO\\^2g%n\x0c3%"
    none_type_1 = None
    var_0 = module_1.dataclass(repr=none_type_1, slots=str_1)
    var_0.__contains__(bytes_0)


def test_case_8():
    none_type_0 = None
    str_0 = "milliseconds"
    dict_0 = {str_0: none_type_0, str_0: str_0}
    user_dict_0 = module_2.UserDict(none_type_0, **dict_0)
    var_0 = user_dict_0.__len__()
    timer_0 = module_0.Timer(text=var_0, logger=none_type_0)
    timer_1 = module_0.Timer(logger=none_type_0)
    timer_2 = timer_1.__enter__()
    timer_3 = module_0.Timer()
    var_1 = timer_1.__call__(timer_1)
    timer_1.__enter__()


def test_case_9():
    timer_error_0 = module_0.TimerError()
    timer_0 = module_0.Timer(timer_error_0)
    none_type_0 = timer_0.start()
    float_0 = timer_0.stop()
    none_type_1 = None
    str_0 = ""
    timer_1 = module_0.Timer(text=str_0, logger=none_type_0)
    list_0 = [none_type_0, timer_error_0, none_type_1, none_type_0]
    var_0 = module_1.dataclass(match_args=none_type_1)
    module_1.dataclass(
        list_0,
        eq=timer_error_0,
        order=var_0,
        unsafe_hash=var_0,
        frozen=list_0,
        match_args=none_type_0,
        kw_only=none_type_0,
        slots=timer_0,
    )


def test_case_10():
    timer_error_0 = module_0.TimerError()
    timer_error_1 = module_0.TimerError()
    float_arg_0 = module_0.FloatArg()
    timer_0 = module_0.Timer(timer_error_0, initial_text=timer_error_1)
    none_type_0 = timer_0.start()
    none_type_1 = None
    float_arg_1 = module_0.FloatArg()
    module_3.Timers(*none_type_1)


def test_case_11():
    none_type_0 = None
    str_0 = "milliseconds"
    dict_0 = {str_0: none_type_0, str_0: str_0}
    user_dict_0 = module_2.UserDict(none_type_0, **dict_0)
    var_0 = user_dict_0.__len__()
    timer_0 = module_0.Timer(text=var_0, logger=none_type_0)
    timer_1 = module_0.Timer(logger=none_type_0)
    timer_2 = user_dict_0.__iter__()
    timer_3 = module_0.Timer()
    var_1 = timer_1.__call__(timer_1)
    timer_4 = timer_1.__enter__()
    none_type_1 = timer_4.__exit__()
    timer_error_0 = module_0.TimerError()
    var_2 = timer_1.__repr__()
    timer_1.__exit__()
