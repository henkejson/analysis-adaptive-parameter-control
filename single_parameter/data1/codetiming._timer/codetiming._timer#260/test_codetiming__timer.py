# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import codetiming._timer as module_0
import dataclasses as module_1
import collections as module_2


def test_case_0():
    timer_error_0 = module_0.TimerError()


def test_case_1():
    timer_0 = module_0.Timer()
    none_type_0 = timer_0.start()


def test_case_2():
    timer_0 = module_0.Timer()
    timer_0.stop()


def test_case_3():
    bool_0 = False
    none_type_0 = None
    timer_0 = module_0.Timer(bool_0, logger=none_type_0)
    var_0 = timer_0.__repr__()
    timer_1 = module_0.Timer(logger=bool_0)
    none_type_1 = timer_1.start()


def test_case_4():
    timer_0 = module_0.Timer()
    float_0 = module_1.dataclass(match_args=timer_0)
    none_type_0 = timer_0.start()
    timer_0.start()


def test_case_5():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    float_0 = timer_1.stop()


def test_case_6():
    timer_0 = module_0.Timer()
    none_type_0 = timer_0.start()
    none_type_1 = timer_0.__exit__()


def test_case_7():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    timer_2 = module_0.Timer(initial_text=timer_1)
    none_type_0 = timer_1.__exit__()
    none_type_1 = timer_2.start()


def test_case_8():
    str_0 = "i;_5K=\\r:)kgaS)6|"
    timer_0 = module_0.Timer(initial_text=str_0)
    timer_1 = module_0.Timer()
    timer_2 = timer_1.__enter__()
    bool_0 = True
    none_type_0 = timer_0.start()
    timer_3 = module_0.Timer(text=str_0)
    var_0 = module_1.dataclass(
        repr=bool_0, unsafe_hash=none_type_0, match_args=timer_1, slots=bool_0
    )
    var_0.__exit__()


def test_case_9():
    none_type_0 = None
    timer_0 = module_0.Timer(none_type_0, none_type_0, logger=none_type_0)
    timer_1 = timer_0.__enter__()
    none_type_1 = timer_0.__exit__()
    timer_0.stop()


def test_case_10():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    str_0 = "F39C>e(5);!%F"
    timer_2 = module_0.Timer(str_0, initial_text=timer_0, logger=timer_1)
    float_0 = timer_1.stop()
    none_type_0 = timer_2.start()
    timer_3 = module_0.Timer()
    timer_4 = module_0.Timer()
    float_arg_0 = module_0.FloatArg()
    list_0 = [none_type_0, none_type_0]
    timer_3.__exit__(*list_0)


def test_case_11():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    str_0 = "F39C>e(5);!%F"
    timer_2 = module_0.Timer(str_0, initial_text=timer_0, logger=timer_1)
    float_0 = timer_1.stop()
    var_0 = timer_0.__eq__(str_0)
    none_type_0 = timer_2.start()
    timer_3 = module_0.Timer()
    timer_4 = module_0.Timer()
    float_arg_0 = module_0.FloatArg()
    str_1 = "Definition of Timer.\n\nSee help(codetiming) for quick instructions, and\nhttps://pypi.org/project/codetiming/ for more details.\n"
    timer_5 = module_0.Timer(none_type_0, none_type_0, str_1)
    float_arg_1 = module_0.FloatArg()
    none_type_1 = timer_2.__exit__()
    var_1 = timer_1.__repr__()
    var_1.start()


def test_case_12():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    float_0 = timer_0.stop()
    bool_0 = True
    complex_0 = 5216.19369 + 1689.1068j
    timer_2 = module_0.Timer(text=timer_1, initial_text=bool_0)
    timer_3 = timer_2.__enter__()
    none_type_0 = timer_3.__exit__()
    module_2.UserDict(complex_0)
