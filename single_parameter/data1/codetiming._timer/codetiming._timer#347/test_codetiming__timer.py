# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import codetiming._timer as module_0


def test_case_0():
    timer_error_0 = module_0.TimerError()


def test_case_1():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()


def test_case_2():
    bool_0 = True
    timer_0 = module_0.Timer(initial_text=bool_0)
    none_type_0 = None
    timer_1 = module_0.Timer(initial_text=bool_0, logger=none_type_0)
    none_type_1 = timer_0.start()
    timer_1.stop()


def test_case_3():
    timer_0 = module_0.Timer()
    none_type_0 = timer_0.start()
    var_0 = timer_0.__eq__(none_type_0)
    float_0 = timer_0.stop()


def test_case_4():
    float_arg_0 = module_0.FloatArg()
    float_arg_1 = module_0.FloatArg()
    none_type_0 = None
    timer_0 = module_0.Timer(none_type_0, none_type_0, logger=none_type_0)
    none_type_1 = timer_0.start()
    none_type_2 = timer_0.__exit__()


def test_case_5():
    bool_0 = True
    timer_0 = module_0.Timer(initial_text=bool_0)
    none_type_0 = timer_0.start()
    timer_0.start()


def test_case_6():
    bool_0 = True
    timer_0 = module_0.Timer(initial_text=bool_0)
    none_type_0 = timer_0.start()


def test_case_7():
    str_0 = "r#2\rd@0u9"
    timer_0 = module_0.Timer(str_0, initial_text=str_0)
    none_type_0 = timer_0.start()
    float_0 = timer_0.stop()


def test_case_8():
    str_0 = "r#2\rd@0u9"
    timer_0 = module_0.Timer(text=str_0)
    timer_1 = module_0.Timer(str_0, initial_text=str_0)
    timer_2 = module_0.Timer(timer_1, timer_1)
    none_type_0 = timer_2.start()
    float_arg_0 = module_0.FloatArg()
    timer_2.stop()


def test_case_9():
    float_arg_0 = module_0.FloatArg()
    bool_0 = True
    timer_0 = module_0.Timer(float_arg_0, initial_text=bool_0)
    none_type_0 = timer_0.start()
    timer_0.start()
