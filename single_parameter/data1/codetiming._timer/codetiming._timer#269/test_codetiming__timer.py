# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import codetiming._timer as module_0


def test_case_0():
    timer_error_0 = module_0.TimerError()


def test_case_1():
    bool_0 = True
    timer_0 = module_0.Timer(bool_0, initial_text=bool_0)
    timer_1 = timer_0.__enter__()


def test_case_2():
    int_0 = 1205
    timer_0 = module_0.Timer(int_0)
    none_type_0 = timer_0.start()
    none_type_1 = timer_0.__exit__()
    list_0 = []
    timer_0.__exit__(*list_0)


def test_case_3():
    int_0 = 1205
    timer_0 = module_0.Timer(int_0)
    none_type_0 = timer_0.start()
    none_type_1 = timer_0.__exit__()


def test_case_4():
    bool_0 = True
    timer_0 = module_0.Timer(bool_0, initial_text=bool_0)
    none_type_0 = timer_0.start()
    timer_0.start()


def test_case_5():
    none_type_0 = None
    timer_error_0 = module_0.TimerError()
    timer_0 = module_0.Timer(none_type_0)
    none_type_1 = timer_0.start()
    list_0 = []
    none_type_2 = timer_0.__exit__(*list_0)
    timer_1 = module_0.Timer()
    timer_2 = module_0.Timer(logger=list_0)
    timer_3 = timer_2.__enter__()
    timer_error_1 = module_0.TimerError()


def test_case_6():
    bool_0 = True
    none_type_0 = None
    timer_0 = module_0.Timer(text=bool_0, initial_text=bool_0, logger=none_type_0)
    timer_error_0 = module_0.TimerError()
    timer_1 = module_0.Timer(timer_0, logger=none_type_0)
    none_type_1 = timer_1.start()
    list_0 = []
    var_0 = timer_1.__eq__(bool_0)
    timer_1.__exit__(*list_0)


def test_case_7():
    str_0 = "P"
    timer_0 = module_0.Timer(text=str_0, initial_text=str_0)
    timer_1 = timer_0.__enter__()
    timer_error_0 = module_0.TimerError()
    timer_0.__enter__()


def test_case_8():
    bool_0 = False
    bool_1 = True
    timer_0 = module_0.Timer(bool_0, initial_text=bool_1)
    none_type_0 = timer_0.start()
    var_0 = timer_0.__eq__(timer_0)
    none_type_1 = timer_0.__exit__()
    timer_0.stop()
