# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import codetiming._timer as module_0


def test_case_0():
    timer_0 = module_0.Timer()


def test_case_1():
    timer_0 = module_0.Timer()
    list_0 = []
    dict_0 = {}
    timer_1 = timer_0.__enter__()
    timer_error_0 = module_0.TimerError(*list_0, **dict_0)
    timer_0.__enter__()


def test_case_2():
    timer_0 = module_0.Timer()
    none_type_0 = timer_0.start()
    none_type_1 = timer_0.__exit__()


def test_case_3():
    timer_0 = module_0.Timer()
    timer_0.__exit__()


def test_case_4():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    none_type_0 = timer_0.__exit__()


def test_case_5():
    timer_error_0 = module_0.TimerError()
    timer_0 = module_0.Timer(initial_text=timer_error_0)
    timer_error_1 = module_0.TimerError()
    none_type_0 = timer_0.start()
    var_0 = timer_0.__eq__(timer_0)
    var_1 = timer_0.__eq__(timer_0)


def test_case_6():
    str_0 = "p*~[ac,^\\1rH4jySxcB9"
    timer_0 = module_0.Timer(initial_text=str_0)
    var_0 = timer_0.__repr__()
    timer_1 = timer_0.__enter__()
    var_1 = timer_0.__call__(str_0)
    timer_0.start()


def test_case_7():
    none_type_0 = None
    timer_0 = module_0.Timer(initial_text=none_type_0, logger=none_type_0)
    timer_1 = timer_0.__enter__()
    float_0 = timer_1.stop()
    var_0 = timer_1.__enter__()
    timer_0.start()


def test_case_8():
    str_0 = "R&/?*?S?5)8^q@I"
    bool_0 = False
    timer_0 = module_0.Timer(str_0, str_0, bool_0)
    timer_1 = timer_0.__enter__()
    float_0 = timer_0.stop()
    timer_1.__exit__()


def test_case_9():
    str_0 = "R&/?*?S?5)8^q@I"
    bool_0 = True
    timer_0 = module_0.Timer(str_0, str_0, bool_0)
    timer_1 = timer_0.__enter__()
    float_0 = timer_0.stop()
    none_type_0 = timer_1.__eq__(str_0)
    none_type_1 = None
    timer_2 = module_0.Timer(text=none_type_1)
    float_arg_0 = module_0.FloatArg()
    timer_3 = module_0.Timer()
    str_1 = "L$v+:bGy[PCc\x0bg-ACV"
    dict_0 = {str_1: timer_3}
    module_0.TimerError(**dict_0)
