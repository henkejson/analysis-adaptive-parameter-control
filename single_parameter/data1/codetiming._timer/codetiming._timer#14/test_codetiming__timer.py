# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import codetiming._timer as module_0


def test_case_0():
    float_arg_0 = module_0.FloatArg()


def test_case_1():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()


def test_case_2():
    str_0 = "o"
    timer_0 = module_0.Timer(str_0, str_0, str_0)
    none_type_0 = timer_0.start()
    float_0 = timer_0.stop()
    var_0 = timer_0.__eq__(str_0)
    timer_0.__exit__()


def test_case_3():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    float_0 = timer_0.stop()


def test_case_4():
    str_0 = "o"
    timer_0 = module_0.Timer(str_0, str_0, str_0)
    none_type_0 = timer_0.start()
    float_0 = timer_0.stop()
    var_0 = timer_0.__repr__()
    none_type_1 = timer_0.start()
    timer_0.__enter__()


def test_case_5():
    none_type_0 = None
    timer_0 = module_0.Timer(logger=none_type_0)
    none_type_1 = timer_0.start()
    float_0 = timer_0.stop()
    var_0 = timer_0.__repr__()
    timer_1 = module_0.Timer()
    timer_1.__exit__(*var_0)


def test_case_6():
    str_0 = "12aV"
    timer_0 = module_0.Timer(str_0, str_0, str_0)
    none_type_0 = timer_0.start()
    float_0 = timer_0.stop()
    var_0 = timer_0.__repr__()
    var_1 = str_0.__len__()
    float_arg_0 = module_0.FloatArg()
    timer_1 = module_0.Timer(initial_text=float_arg_0, logger=var_0)
    timer_1.start()


def test_case_7():
    str_0 = "12aV"
    timer_0 = module_0.Timer(str_0, str_0, str_0)
    var_0 = timer_0.__repr__()
    none_type_0 = timer_0.start()
    float_0 = timer_0.stop()
    var_1 = timer_0.__repr__()
    var_2 = str_0.__len__()
    float_arg_0 = module_0.FloatArg()
    timer_1 = module_0.Timer(initial_text=float_arg_0, logger=var_1)
    timer_2 = module_0.Timer(str_0, initial_text=float_arg_0)
    timer_3 = timer_2.__enter__()


def test_case_8():
    str_0 = "12aV"
    timer_0 = module_0.Timer(str_0, str_0, str_0)
    timer_1 = module_0.Timer(text=timer_0, initial_text=timer_0, logger=timer_0)
    none_type_0 = timer_1.start()
    float_0 = timer_1.stop()
    var_0 = timer_1.__repr__()
    var_1 = var_0.__len__()
    timer_2 = module_0.Timer(text=str_0)
    none_type_1 = timer_2.start()
    var_0.__exit__()
