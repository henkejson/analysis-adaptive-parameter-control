# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import codetiming._timer as module_0
import collections as module_1
import dataclasses as module_2


def test_case_0():
    timer_error_0 = module_0.TimerError()


def test_case_1():
    timer_error_0 = module_0.TimerError()
    timer_0 = module_0.Timer(initial_text=timer_error_0)
    timer_1 = timer_0.__enter__()


def test_case_2():
    str_0 = "$9<"
    timer_0 = module_0.Timer(initial_text=str_0, logger=str_0)
    timer_0.__exit__()


def test_case_3():
    bool_0 = True
    timer_0 = module_0.Timer(initial_text=bool_0)
    float_arg_0 = module_0.FloatArg()
    var_0 = timer_0.__eq__(bool_0)
    timer_1 = timer_0.__enter__()
    none_type_0 = timer_0.__exit__()
    user_dict_0 = module_1.UserDict()
    user_dict_0.__contains__(timer_1)


def test_case_4():
    timer_0 = module_0.Timer()
    none_type_0 = timer_0.start()
    float_0 = timer_0.stop()


def test_case_5():
    str_0 = "$9<"
    timer_0 = module_0.Timer(initial_text=str_0, logger=str_0)
    var_0 = timer_0.__repr__()
    timer_0.__enter__()


def test_case_6():
    bool_0 = True
    timer_0 = module_0.Timer(initial_text=bool_0)
    var_0 = timer_0.__call__(bool_0)
    timer_1 = timer_0.__enter__()
    timer_0.__enter__()


def test_case_7():
    none_type_0 = None
    bytes_0 = b"\xc1j\x07\xa1\x82h\xf9|1t,"
    timer_0 = module_0.Timer(text=bytes_0, logger=none_type_0)
    var_0 = timer_0.__repr__()
    none_type_1 = timer_0.start()
    var_1 = module_2.field(init=none_type_0)
    var_1.max(none_type_1)


def test_case_8():
    bool_0 = True
    timer_0 = module_0.Timer(bool_0)
    var_0 = timer_0.__call__(timer_0)
    timer_1 = timer_0.__enter__()
    none_type_0 = timer_0.__exit__()


def test_case_9():
    none_type_0 = None
    timer_0 = module_0.Timer(initial_text=none_type_0, logger=none_type_0)
    timer_1 = timer_0.__enter__()
    none_type_1 = timer_0.__exit__()
    timer_1.__delitem__(timer_1)


def test_case_10():
    bool_0 = True
    timer_0 = module_0.Timer(bool_0, bool_0, bool_0, bool_0)
    timer_0.__enter__()


def test_case_11():
    bool_0 = True
    timer_0 = module_0.Timer(initial_text=bool_0)
    float_arg_0 = module_0.FloatArg()
    timer_1 = module_0.Timer(text=timer_0)
    timer_2 = timer_1.__enter__()
    none_type_0 = timer_1.__exit__()
    var_0 = timer_1.__eq__(timer_2)
    var_0.__contains__(none_type_0)
