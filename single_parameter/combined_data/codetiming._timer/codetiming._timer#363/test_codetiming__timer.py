# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import codetiming._timer as module_0
import collections as module_1


def test_case_0():
    timer_error_0 = module_0.TimerError()


def test_case_1():
    timer_0 = module_0.Timer()
    none_type_0 = timer_0.start()
    none_type_1 = timer_0.__exit__()


def test_case_2():
    timer_0 = module_0.Timer()
    list_0 = [timer_0, timer_0, timer_0]
    timer_error_0 = module_0.TimerError(*list_0)
    timer_1 = module_0.Timer(timer_0, initial_text=timer_0)
    none_type_0 = timer_1.__eq__(list_0)
    timer_2 = module_0.Timer(logger=timer_1)
    var_0 = timer_2.__eq__(none_type_0)
    none_type_1 = timer_1.start()
    timer_2.__exit__()


def test_case_3():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    list_0 = [timer_0, timer_0, timer_1]
    timer_error_0 = module_0.TimerError(*list_0)
    timer_2 = module_0.Timer(timer_0, initial_text=timer_0)
    none_type_0 = timer_2.start()
    timer_1.start()


def test_case_4():
    timer_error_0 = module_0.TimerError()
    str_0 = "b9A~!x"
    timer_0 = module_0.Timer(str_0, str_0, str_0)
    none_type_0 = timer_0.start()
    str_1 = "kSqJQu%eW:"
    str_2 = "Timer is running. Use .stop() to stop it"
    dict_0 = {str_1: none_type_0, str_0: none_type_0, str_2: none_type_0}
    module_0.TimerError(**dict_0)


def test_case_5():
    none_type_0 = None
    timer_0 = module_0.Timer(logger=none_type_0)
    timer_error_0 = module_0.TimerError()
    timer_1 = timer_0.__enter__()


def test_case_6():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    var_0 = timer_0.__repr__()
    float_0 = timer_0.stop()
    none_type_0 = timer_0.start()
    none_type_1 = timer_0.__exit__()
    var_1 = timer_1.__eq__(none_type_1)
    timer_2 = timer_0.__enter__()
    timer_1.start()


def test_case_7():
    timer_0 = module_0.Timer()
    list_0 = [timer_0, timer_0]
    timer_error_0 = module_0.TimerError(*list_0)
    str_0 = "{vKe&"
    timer_1 = module_0.Timer(text=str_0, initial_text=timer_0)
    none_type_0 = timer_1.start()
    none_type_1 = timer_0.start()
    user_dict_0 = module_1.UserDict()
    var_0 = user_dict_0.__ror__(none_type_1)
    var_0.__ior__(none_type_0)


def test_case_8():
    timer_error_0 = module_0.TimerError()
    str_0 = "b9A~!x"
    timer_0 = module_0.Timer(str_0, str_0, str_0)
    none_type_0 = timer_0.start()
    str_1 = "kSqJQu%eW:"
    str_2 = "Timer is running. Use .stop() to stop it"
    dict_0 = {str_1: none_type_0, str_0: none_type_0, str_2: none_type_0}
    none_type_1 = timer_0.__exit__()
    module_0.TimerError(**dict_0)


def test_case_9():
    str_0 = "1iJp-ku"
    none_type_0 = None
    timer_0 = module_0.Timer(logger=none_type_0)
    none_type_1 = timer_0.start()
    str_1 = "1u!&,]>Z.nDQ\x0bO*&"
    float_0 = timer_0.stop()
    user_dict_0 = module_1.UserDict()
    timer_1 = module_0.Timer(text=none_type_1, logger=str_0)
    timer_2 = module_0.Timer(none_type_0, initial_text=str_1, logger=none_type_0)
    timer_0.__exit__()
