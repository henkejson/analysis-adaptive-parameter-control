# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import codetiming._timer as module_0
import collections as module_1


def test_case_0():
    float_arg_0 = module_0.FloatArg()


def test_case_1():
    none_type_0 = None
    timer_0 = module_0.Timer(logger=none_type_0)
    none_type_1 = timer_0.start()
    timer_0.start()


def test_case_2():
    timer_0 = module_0.Timer()
    var_0 = timer_0.__enter__()


def test_case_3():
    timer_0 = module_0.Timer()
    timer_0.__exit__()


def test_case_4():
    timer_0 = module_0.Timer()
    none_type_0 = timer_0.start()
    float_0 = timer_0.stop()


def test_case_5():
    none_type_0 = None
    timer_0 = module_0.Timer(initial_text=none_type_0, logger=none_type_0)
    none_type_1 = timer_0.__repr__()
    timer_error_0 = module_0.TimerError()
    timer_error_1 = module_0.TimerError()
    none_type_2 = timer_0.start()
    none_type_3 = None
    float_arg_0 = module_0.FloatArg()
    list_0 = [none_type_3, none_type_3, none_type_3, none_type_3]
    timer_error_2 = module_0.TimerError()
    float_0 = timer_0.stop()
    module_0.FloatArg(*list_0)


def test_case_6():
    float_arg_0 = module_0.FloatArg()
    bytes_0 = b"\xd1\xab\x00\xcf\x95"
    timer_0 = module_0.Timer(initial_text=bytes_0)
    timer_1 = timer_0.__enter__()


def test_case_7():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    float_arg_0 = module_0.FloatArg()
    float_0 = timer_0.stop()
    str_0 = "?8n= ;-3^I\t3"
    user_dict_0 = module_1.UserDict()
    var_0 = timer_1.__call__(timer_0)
    timer_2 = module_0.Timer()
    timer_3 = module_0.Timer(text=str_0, logger=var_0)
    timer_4 = module_0.Timer(var_0)
    timer_5 = module_0.Timer(var_0, var_0, var_0)
    var_1 = user_dict_0.__len__()
    timer_6 = timer_5.__enter__()
    var_0.__contains__(user_dict_0)


def test_case_8():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    float_arg_0 = module_0.FloatArg()
    float_0 = timer_0.stop()
    timer_2 = timer_0.__enter__()
    int_0 = 0
    str_0 = "?8n= ;-3^I\t3"
    user_dict_0 = module_1.UserDict()
    var_0 = timer_0.__eq__(int_0)
    var_1 = timer_1.__call__(int_0)
    timer_3 = module_0.Timer()
    timer_4 = module_0.Timer(text=str_0, logger=var_1)
    timer_5 = module_0.Timer(var_1)
    timer_6 = module_0.Timer(int_0, var_1, var_1)
    var_2 = user_dict_0.__len__()
    timer_7 = timer_6.__enter__()
    float_arg_1 = module_0.FloatArg()
    none_type_0 = timer_5.start()
    none_type_1 = timer_4.start()
    timer_7.stop()


def test_case_9():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    float_arg_0 = module_0.FloatArg()
    float_0 = timer_0.stop()
    timer_2 = timer_0.__enter__()
    int_0 = 0
    str_0 = "?8n ;-3'I\t3"
    user_dict_0 = module_1.UserDict()
    var_0 = timer_1.__call__(int_0)
    timer_3 = module_0.Timer()
    timer_4 = module_0.Timer(initial_text=str_0)
    timer_5 = module_0.Timer(logger=str_0)
    str_1 = "Lu::;vgtvN"
    timer_6 = module_0.Timer(str_1, logger=var_0)
    var_1 = user_dict_0.__len__()
    timer_7 = timer_4.__enter__()
    module_0.FloatArg(*str_1, **var_1)


def test_case_10():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    float_arg_0 = module_0.FloatArg()
    float_0 = timer_0.stop()
    var_0 = timer_0.__repr__()
    int_0 = 0
    str_0 = "?8n= ;-3^I\t3"
    user_dict_0 = module_1.UserDict()
    var_1 = timer_1.__call__(int_0)
    timer_2 = module_0.Timer()
    timer_3 = module_0.Timer(text=str_0, logger=var_1)
    timer_4 = module_0.Timer(var_1)
    timer_5 = module_0.Timer(int_0, var_1, var_1)
    var_2 = user_dict_0.__len__()
    timer_6 = timer_5.__enter__()
    float_arg_1 = module_0.FloatArg()
    none_type_0 = timer_4.start()
    timer_error_0 = module_0.TimerError()
    float_1 = timer_4.stop()
    module_0.FloatArg(*var_0)
