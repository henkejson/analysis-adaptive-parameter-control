# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import codetiming._timer as module_0


def test_case_0():
    timer_0 = module_0.Timer()


def test_case_1():
    timer_0 = module_0.Timer()
    none_type_0 = timer_0.start()


def test_case_2():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    float_arg_0 = module_0.FloatArg()
    var_0 = timer_0.__repr__()
    var_1 = var_0.__iter__()
    float_0 = timer_1.stop()
    timer_0.__exit__()


def test_case_3():
    timer_0 = module_0.Timer()
    none_type_0 = timer_0.start()
    none_type_1 = timer_0.__exit__()


def test_case_4():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    float_arg_0 = module_0.FloatArg()
    float_0 = timer_1.stop()
    var_0 = timer_0.__repr__()
    timer_2 = timer_0.__enter__()
    timer_0.start()


def test_case_5():
    timer_0 = module_0.Timer()
    none_type_0 = timer_0.start()
    float_0 = timer_0.stop()


def test_case_6():
    float_arg_0 = module_0.FloatArg()
    timer_0 = module_0.Timer(initial_text=float_arg_0)
    timer_1 = timer_0.__enter__()


def test_case_7():
    float_arg_0 = module_0.FloatArg()
    bool_0 = True
    timer_0 = module_0.Timer(initial_text=bool_0)
    none_type_0 = timer_0.start()
    timer_1 = module_0.Timer(none_type_0, logger=none_type_0)
    none_type_1 = timer_0.__exit__()
    timer_2 = timer_1.__enter__()
    float_0 = timer_1.stop()
    var_0 = timer_0.__call__(none_type_0)
    timer_3 = module_0.Timer(initial_text=bool_0)
    var_0.__contains__(none_type_1)


def test_case_8():
    float_arg_0 = module_0.FloatArg()
    timer_0 = module_0.Timer(float_arg_0, float_arg_0, float_arg_0)
    none_type_0 = timer_0.start()
    timer_0.__exit__()


def test_case_9():
    str_0 = 'Ma"3%{}( 0Hc<}D!L#?'
    timer_0 = module_0.Timer(text=str_0, initial_text=str_0, logger=str_0)
    timer_0.start()


def test_case_10():
    float_arg_0 = module_0.FloatArg()
    bool_0 = True
    timer_0 = module_0.Timer(initial_text=bool_0)
    none_type_0 = timer_0.start()
    none_type_1 = None
    timer_1 = module_0.Timer(bool_0, logger=none_type_1)
    none_type_2 = timer_0.__exit__()
    timer_2 = timer_1.__enter__()
    float_0 = timer_1.stop()
    var_0 = timer_0.__call__(none_type_0)
    timer_3 = module_0.Timer(initial_text=bool_0)
    var_0.__contains__(none_type_2)
