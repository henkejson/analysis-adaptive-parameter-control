# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import codetiming._timer as module_0


def test_case_0():
    timer_error_0 = module_0.TimerError()


def test_case_1():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()


def test_case_2():
    timer_0 = module_0.Timer()
    timer_0.stop()


def test_case_3():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    none_type_0 = timer_0.__exit__()


def test_case_4():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    none_type_0 = timer_0.__exit__()
    timer_2 = module_0.Timer(none_type_0, initial_text=timer_1, logger=none_type_0)
    var_0 = timer_1.__eq__(timer_2)
    timer_3 = timer_0.__enter__()
    var_1 = timer_0.__eq__(timer_0)
    timer_4 = timer_2.__enter__()
    var_2 = timer_0.__call__(timer_4)
    timer_1.start()


def test_case_5():
    timer_0 = module_0.Timer()
    var_0 = timer_0.__call__(timer_0)
    timer_1 = timer_0.__enter__()
    none_type_0 = timer_0.__exit__()
    timer_2 = module_0.Timer(none_type_0, initial_text=timer_1, logger=var_0)
    timer_3 = timer_0.__enter__()
    var_1 = timer_0.__eq__(timer_0)
    timer_2.__enter__()


def test_case_6():
    str_0 = "H%Vs*\\%oq"
    timer_0 = module_0.Timer(str_0, str_0)
    none_type_0 = timer_0.start()
    float_0 = timer_0.stop()
    str_1 = "K4ur)F"
    timer_1 = module_0.Timer(text=str_0)
    dict_0 = {str_0: str_0, str_0: str_0, str_1: str_0, str_0: str_1}
    timer_2 = timer_0.__enter__()
    timer_3 = timer_1.__enter__()
    var_0 = timer_0.__eq__(timer_0)
    module_0.TimerError(**dict_0)


def test_case_7():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    var_0 = timer_1.__repr__()
    none_type_0 = timer_0.__exit__()
    timer_2 = module_0.Timer(none_type_0, initial_text=var_0, logger=timer_1)
    var_1 = timer_1.__eq__(var_0)
    var_2 = timer_0.__eq__(timer_0)
    timer_2.__enter__()


def test_case_8():
    timer_0 = module_0.Timer()
    var_0 = timer_0.__repr__()
    timer_1 = timer_0.__enter__()
    none_type_0 = timer_0.__exit__()
    timer_2 = module_0.Timer(none_type_0, initial_text=timer_1, logger=none_type_0)
    var_1 = timer_1.__eq__(var_0)
    timer_3 = timer_0.__enter__()
    var_2 = timer_0.__eq__(timer_0)
    timer_4 = timer_2.__enter__()
    float_0 = timer_4.stop()
    var_2.__ior__(none_type_0)


def test_case_9():
    timer_0 = module_0.Timer()
    var_0 = timer_0.__repr__()
    timer_1 = timer_0.__enter__()
    none_type_0 = timer_0.__exit__()
    timer_2 = module_0.Timer(none_type_0, initial_text=timer_1, logger=var_0)
    timer_2.__enter__()


def test_case_10():
    timer_0 = module_0.Timer()
    var_0 = timer_0.__repr__()
    timer_1 = module_0.Timer(var_0, initial_text=timer_0, logger=var_0)
    timer_2 = timer_0.__enter__()
    var_1 = timer_0.__eq__(timer_0)
    timer_1.__enter__()
