# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import codetiming._timer as module_0
import collections as module_1


def test_case_0():
    float_arg_0 = module_0.FloatArg()


def test_case_1():
    str_0 = "D\\@5.%8"
    timer_0 = module_0.Timer(str_0, initial_text=str_0)
    timer_1 = timer_0.__enter__()


def test_case_2():
    timer_0 = module_0.Timer()
    timer_0.__exit__()


def test_case_3():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    float_0 = timer_0.stop()


def test_case_4():
    timer_0 = module_0.Timer()
    var_0 = timer_0.__enter__()
    timer_0.__enter__()


def test_case_5():
    bool_0 = False
    timer_0 = module_0.Timer(logger=bool_0)
    timer_1 = timer_0.__enter__()
    var_0 = timer_0.__repr__()
    timer_2 = module_0.Timer()
    none_type_0 = timer_0.__exit__()
    dict_0 = {}
    none_type_1 = timer_0.start()
    module_0.TimerError(*bool_0, **dict_0)


def test_case_6():
    str_0 = "\\f83u\x0cu-*LZ\x0bjDx<uz"
    str_1 = " does not support item assignment. Use '.add()' to update values."
    timer_0 = module_0.Timer(str_0, str_1, str_1)
    float_arg_0 = module_0.FloatArg()
    none_type_0 = timer_0.start()
    none_type_1 = timer_0.__exit__()
    timer_0.__exit__()


def test_case_7():
    bool_0 = True
    timer_0 = module_0.Timer(initial_text=bool_0)
    none_type_0 = timer_0.start()


def test_case_8():
    float_arg_0 = module_0.FloatArg()
    list_0 = []
    timer_0 = module_0.Timer(float_arg_0, initial_text=float_arg_0)
    none_type_0 = timer_0.start()
    none_type_1 = None
    timer_1 = module_0.Timer(none_type_1, logger=none_type_1)
    user_dict_0 = module_1.UserDict(list_0)
    float_0 = timer_0.stop()
    var_0 = user_dict_0.items()
    var_0.apply(var_0, var_0)
