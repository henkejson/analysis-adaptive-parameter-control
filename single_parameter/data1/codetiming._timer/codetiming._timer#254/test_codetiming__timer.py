# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import codetiming._timer as module_0


def test_case_0():
    float_arg_0 = module_0.FloatArg()


def test_case_1():
    str_0 = "HEzkS.]J|}<#'G,=L"
    timer_0 = module_0.Timer(str_0)
    timer_1 = timer_0.__enter__()
    timer_1.__enter__()


def test_case_2():
    str_0 = "seconds"
    timer_0 = module_0.Timer(initial_text=str_0)
    timer_1 = timer_0.__enter__()


def test_case_3():
    str_0 = "HEzkS.]J|}<#'G,=L"
    timer_0 = module_0.Timer(str_0)
    timer_1 = timer_0.__enter__()
    none_type_0 = timer_1.__exit__()
    timer_0.__exit__()


def test_case_4():
    str_0 = "HEzkS.]J|}<#'G,=L"
    timer_0 = module_0.Timer(str_0)
    timer_1 = timer_0.__enter__()
    none_type_0 = timer_1.__exit__()


def test_case_5():
    str_0 = "seconds"
    timer_0 = module_0.Timer(initial_text=str_0)
    timer_1 = timer_0.__enter__()
    float_0 = timer_1.stop()


def test_case_6():
    str_0 = "HEzkS.]J|}<#'G,=L"
    none_type_0 = None
    timer_0 = module_0.Timer(str_0, logger=none_type_0)
    timer_1 = timer_0.__enter__()
    float_0 = timer_0.stop()
    timer_1.__exit__()


def test_case_7():
    float_arg_0 = module_0.FloatArg()
    complex_0 = -40.93132 + 489.6244j
    timer_0 = module_0.Timer()
    none_type_0 = timer_0.start()
    var_0 = timer_0.__repr__()
    timer_1 = module_0.Timer(var_0, timer_0)
    float_arg_1 = module_0.FloatArg()
    var_1 = timer_0.__call__(complex_0)
    none_type_1 = timer_0.__exit__()
    timer_2 = timer_1.__enter__()
    none_type_2 = timer_2.__exit__()
    var_0.__exit__()


def test_case_8():
    str_0 = "@+4K"
    timer_0 = module_0.Timer(initial_text=str_0)
    timer_1 = module_0.Timer(initial_text=timer_0)
    timer_2 = timer_1.__enter__()
    timer_0.__exit__()


def test_case_9():
    str_0 = "@+4K"
    timer_0 = module_0.Timer(initial_text=str_0)
    float_arg_0 = module_0.FloatArg()
    timer_1 = module_0.Timer(str_0, initial_text=timer_0)
    timer_2 = timer_1.__enter__()
    var_0 = timer_1.__repr__()
    var_1 = var_0.__eq__(var_0)
    timer_0.stop()
