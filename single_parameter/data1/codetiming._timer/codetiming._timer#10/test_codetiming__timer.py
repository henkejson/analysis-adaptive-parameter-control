# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import codetiming._timer as module_0


def test_case_0():
    timer_error_0 = module_0.TimerError()


def test_case_1():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__call__(timer_0)
    timer_2 = timer_0.__enter__()


def test_case_2():
    timer_0 = module_0.Timer()
    timer_0.stop()


def test_case_3():
    none_type_0 = None
    timer_0 = module_0.Timer(text=none_type_0, logger=none_type_0)
    var_0 = timer_0.__call__(none_type_0)
    var_0.__call__(none_type_0)


def test_case_4():
    float_arg_0 = module_0.FloatArg()
    timer_0 = module_0.Timer()
    timer_0.__exit__()


def test_case_5():
    none_type_0 = None
    timer_0 = module_0.Timer(text=none_type_0, logger=none_type_0)
    var_0 = timer_0.__call__(none_type_0)
    timer_1 = timer_0.__enter__()
    var_0.__call__(none_type_0)


def test_case_6():
    none_type_0 = None
    timer_0 = module_0.Timer(text=none_type_0, logger=none_type_0)
    var_0 = timer_0.__eq__(none_type_0)
    timer_1 = module_0.Timer(initial_text=var_0)
    var_1 = timer_1.__call__(none_type_0)
    var_1.__call__(timer_0)


def test_case_7():
    none_type_0 = None
    timer_0 = module_0.Timer(text=none_type_0, logger=none_type_0)
    timer_1 = module_0.Timer(initial_text=timer_0)
    timer_2 = timer_1.__enter__()
    var_0 = timer_2.__repr__()
    var_1 = timer_1.__call__(timer_0)
    var_0.get(var_0)


def test_case_8():
    none_type_0 = None
    timer_0 = module_0.Timer(text=none_type_0, logger=none_type_0)
    var_0 = timer_0.__repr__()
    timer_1 = module_0.Timer(initial_text=var_0)
    var_1 = timer_1.__call__(none_type_0)
    timer_2 = module_0.Timer(logger=timer_0)
    var_1.__call__(timer_0)


def test_case_9():
    none_type_0 = None
    timer_0 = module_0.Timer(logger=none_type_0)
    timer_1 = timer_0.__enter__()
    timer_2 = module_0.Timer(timer_1, initial_text=none_type_0, logger=none_type_0)
    var_0 = timer_2.__call__(none_type_0)
    var_0.__call__(timer_1)


def test_case_10():
    none_type_0 = None
    timer_0 = module_0.Timer(text=none_type_0, logger=none_type_0)
    var_0 = timer_0.__call__(none_type_0)
    timer_1 = module_0.Timer(var_0, initial_text=timer_0)
    var_1 = timer_1.__call__(none_type_0)
    var_1.__call__(var_1)


def test_case_11():
    none_type_0 = None
    timer_0 = module_0.Timer(text=none_type_0, logger=none_type_0)
    timer_1 = timer_0.__enter__()
    timer_2 = module_0.Timer(text=timer_1)
    var_0 = timer_2.__call__(none_type_0)
    var_0.__call__(timer_1)
