# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import codetiming._timer as module_0
import collections as module_1
import dataclasses as module_2


def test_case_0():
    timer_0 = module_0.Timer()


def test_case_1():
    str_0 = "NoPy"
    none_type_0 = None
    timer_0 = module_0.Timer(str_0, logger=none_type_0)
    timer_1 = timer_0.__enter__()
    timer_1.__enter__()


def test_case_2():
    timer_0 = module_0.Timer()
    none_type_0 = timer_0.start()
    none_type_1 = timer_0.__exit__()


def test_case_3():
    str_0 = "CLzo^Sm2 "
    timer_0 = module_0.Timer(str_0, initial_text=str_0, logger=str_0)
    timer_0.stop()


def test_case_4():
    str_0 = "N\\ogy"
    timer_0 = module_0.Timer(initial_text=str_0)
    timer_1 = timer_0.__enter__()
    timer_error_0 = module_0.TimerError()


def test_case_5():
    none_type_0 = None
    float_arg_0 = module_0.FloatArg()
    bool_0 = False
    timer_0 = module_0.Timer(text=bool_0, logger=none_type_0)
    timer_1 = timer_0.__enter__()
    timer_error_0 = module_0.TimerError()
    timer_2 = module_0.Timer(text=none_type_0)
    var_0 = timer_1.__eq__(float_arg_0)
    timer_3 = module_0.Timer(logger=none_type_0)
    user_dict_0 = module_1.UserDict()
    float_0 = timer_1.stop()
    var_1 = none_type_0.__eq__(var_0)
    var_2 = user_dict_0.__eq__(none_type_0)
    user_dict_0.__delitem__(var_2)


def test_case_6():
    str_0 = "-n2;Is%DqyMpI"
    bool_0 = True
    timer_0 = module_0.Timer(text=str_0, initial_text=bool_0)
    var_0 = timer_0.__repr__()
    timer_1 = timer_0.__enter__()
    timer_error_0 = module_0.TimerError(*var_0)
    var_1 = timer_0.__eq__(var_0)
    var_1.__iter__()


def test_case_7():
    str_0 = "-n2;Is%DqyMpI"
    none_type_0 = None
    timer_0 = module_0.Timer(str_0, str_0, none_type_0)
    timer_1 = timer_0.__enter__()
    bool_0 = True
    float_0 = timer_0.__call__(timer_1)
    none_type_1 = timer_1.__exit__()
    none_type_2 = timer_1.start()
    timer_2 = module_0.Timer(text=str_0, initial_text=bool_0)
    var_0 = timer_2.__repr__()
    var_0.__setitem__(float_0, timer_2)


def test_case_8():
    float_arg_0 = module_0.FloatArg()
    timer_0 = module_0.Timer(float_arg_0, initial_text=float_arg_0)
    none_type_0 = timer_0.start()
    var_0 = module_2.dataclass(eq=float_arg_0, unsafe_hash=float_arg_0)
    var_0.start()
