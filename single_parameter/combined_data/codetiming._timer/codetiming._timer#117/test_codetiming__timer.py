# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import codetiming._timer as module_0
import collections as module_1


def test_case_0():
    timer_0 = module_0.Timer()


def test_case_1():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    float_0 = timer_1.stop()


def test_case_2():
    timer_0 = module_0.Timer()
    timer_0.stop()


def test_case_3():
    bool_0 = False
    timer_0 = module_0.Timer(bool_0, initial_text=bool_0)
    timer_1 = timer_0.__enter__()
    none_type_0 = timer_0.__exit__()


def test_case_4():
    timer_0 = module_0.Timer()
    none_type_0 = timer_0.start()
    bytes_0 = b"L\x12\x82`Y\xa9"
    timer_1 = module_0.Timer(text=bytes_0, logger=none_type_0)
    var_0 = timer_0.__repr__()
    timer_2 = timer_1.__enter__()
    none_type_1 = None
    module_0.FloatArg(*none_type_1)


def test_case_5():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    timer_error_0 = module_0.TimerError()
    timer_0.__enter__()


def test_case_6():
    bool_0 = True
    timer_0 = module_0.Timer(bool_0, initial_text=bool_0)
    var_0 = timer_0.__enter__()
    none_type_0 = timer_0.__exit__()


def test_case_7():
    bool_0 = True
    list_0 = [bool_0]
    timer_0 = module_0.Timer()
    none_type_0 = timer_0.start()
    bytes_0 = b"L\x12\x82`Y\xa9"
    timer_1 = module_0.Timer(text=bytes_0, logger=none_type_0)
    var_0 = timer_0.__repr__()
    timer_2 = timer_1.__enter__()
    none_type_1 = timer_1.__exit__()
    var_1 = list_0.__contains__(none_type_1)
    none_type_2 = None
    module_0.FloatArg(*none_type_2)


def test_case_8():
    bool_0 = True
    none_type_0 = None
    timer_0 = module_0.Timer(none_type_0, none_type_0, bool_0)
    list_0 = [none_type_0]
    timer_error_0 = module_0.TimerError(*list_0)
    timer_1 = timer_0.__enter__()
    var_0 = timer_1.__eq__(list_0)
    var_1 = timer_0.__repr__()
    var_1.__getitem__(list_0)


def test_case_9():
    none_type_0 = None
    str_0 = "hp(_kuAQV:8C"
    timer_0 = module_0.Timer(initial_text=str_0)
    user_dict_0 = module_1.UserDict()
    none_type_1 = timer_0.start()
    var_0 = user_dict_0.keys()
    var_0.__getitem__(none_type_0)


def test_case_10():
    bool_0 = True
    timer_0 = module_0.Timer(bool_0, initial_text=bool_0)
    timer_1 = module_0.Timer(text=timer_0)
    timer_2 = timer_1.__enter__()
    var_0 = timer_1.__repr__()
    none_type_0 = timer_1.__exit__()
    var_0.__delitem__(var_0)
