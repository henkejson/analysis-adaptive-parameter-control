# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import codetiming._timer as module_0


def test_case_0():
    timer_error_0 = module_0.TimerError()


def test_case_1():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    float_0 = timer_1.stop()
    timer_0.stop()


def test_case_2():
    bool_0 = True
    timer_0 = module_0.Timer(initial_text=bool_0)
    timer_1 = timer_0.__enter__()
    none_type_0 = timer_1.__exit__()


def test_case_3():
    bytes_0 = b"\xb0\xd1\x02\xb1\x97I\xff`"
    timer_0 = module_0.Timer(logger=bytes_0)
    none_type_0 = timer_0.start()


def test_case_4():
    bool_0 = True
    timer_0 = module_0.Timer(initial_text=bool_0)
    timer_1 = timer_0.__enter__()


def test_case_5():
    int_0 = 117
    str_0 = " does not support item assignment. Use '.add()' to update values."
    timer_0 = module_0.Timer(text=int_0, initial_text=str_0, logger=int_0)
    timer_0.__enter__()


def test_case_6():
    bytes_0 = b"\xb0\xd1\x02\xb1\x97I\xff`"
    timer_0 = module_0.Timer(logger=bytes_0)
    timer_1 = timer_0.__enter__()
    timer_1.start()


def test_case_7():
    none_type_0 = None
    timer_0 = module_0.Timer(logger=none_type_0)
    none_type_1 = timer_0.start()


def test_case_8():
    bytes_0 = b"\xb0\xd1\x02\xb1\x97I\xff`"
    timer_0 = module_0.Timer(logger=bytes_0)
    bool_0 = True
    timer_1 = module_0.Timer(bool_0, logger=timer_0)
    timer_2 = timer_1.__enter__()
    none_type_0 = timer_2.__exit__()
    timer_3 = timer_2.__enter__()
    timer_3.__enter__()


def test_case_9():
    none_type_0 = None
    timer_0 = module_0.Timer(logger=none_type_0)
    bool_0 = True
    timer_1 = module_0.Timer(initial_text=bool_0)
    var_0 = timer_1.__repr__()
    timer_2 = timer_1.__enter__()
    none_type_1 = timer_2.__exit__()
    timer_3 = timer_0.__enter__()
    float_0 = timer_3.stop()
    none_type_2 = timer_2.start()


def test_case_10():
    bytes_0 = b"\xb0\xd1\x02\xb1\x97I\xff`"
    timer_0 = module_0.Timer(logger=bytes_0)
    bool_0 = True
    timer_1 = module_0.Timer(timer_0, initial_text=bool_0)
    var_0 = timer_0.__repr__()
    timer_2 = timer_1.__enter__()
    var_0.__exit__()


def test_case_11():
    bytes_0 = b"\xb0\xd1\x02\xb1\x97I\xff`"
    timer_0 = module_0.Timer(logger=bytes_0)
    none_type_0 = None
    timer_1 = module_0.Timer(none_type_0, timer_0)
    var_0 = timer_0.__repr__()
    timer_2 = timer_1.__enter__()
    none_type_1 = timer_2.__exit__()
    timer_3 = timer_2.__enter__()
    timer_1.__enter__()
