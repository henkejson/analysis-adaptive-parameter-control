# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import codetiming._timer as module_0


def test_case_0():
    float_arg_0 = module_0.FloatArg()


def test_case_1():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()


def test_case_2():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    list_0 = [timer_0]
    none_type_0 = timer_1.__exit__(*list_0)
    timer_1.__exit__()


def test_case_3():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    none_type_0 = timer_0.__exit__()


def test_case_4():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    none_type_0 = None
    timer_2 = module_0.Timer(initial_text=none_type_0, logger=none_type_0)
    none_type_1 = timer_2.start()
    list_0 = [timer_1, timer_0, none_type_1]
    none_type_2 = timer_0.__exit__(*list_0)
    var_0 = timer_1.__eq__(timer_1)
    var_0.__len__()


def test_case_5():
    timer_0 = module_0.Timer()
    timer_1 = module_0.Timer(initial_text=timer_0, logger=timer_0)
    none_type_0 = timer_1.start()
    timer_1.__enter__()


def test_case_6():
    timer_0 = module_0.Timer()
    none_type_0 = None
    timer_1 = module_0.Timer(initial_text=none_type_0, logger=none_type_0)
    none_type_1 = timer_1.start()
    list_0 = [timer_1, timer_0, none_type_1]
    none_type_2 = timer_1.__exit__(*list_0)
    timer_0.__exit__(*list_0)


def test_case_7():
    timer_0 = module_0.Timer()
    var_0 = timer_0.__call__(timer_0)
    timer_1 = module_0.Timer(initial_text=var_0, logger=var_0)
    none_type_0 = timer_1.start()
    var_0.__setitem__(timer_0, timer_0)


def test_case_8():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__repr__()
    timer_2 = module_0.Timer(initial_text=timer_1, logger=timer_1)
    timer_2.start()


def test_case_9():
    timer_0 = module_0.Timer()
    var_0 = timer_0.__call__(timer_0)
    var_1 = timer_0.__call__(timer_0)
    timer_1 = module_0.Timer(timer_0, timer_0, var_1, var_1)
    none_type_0 = timer_1.start()
    list_0 = [var_0, timer_0, var_0, none_type_0]
    timer_0.__exit__(*list_0)


def test_case_10():
    bool_0 = True
    str_0 = "7?o@h=pJBKCpK5j+H"
    none_type_0 = None
    timer_0 = module_0.Timer(str_0, str_0, none_type_0)
    timer_1 = module_0.Timer()
    none_type_1 = timer_0.start()
    var_0 = timer_0.__eq__(bool_0)
    var_1 = timer_0.__call__(none_type_1)
    float_0 = timer_0.stop()
    var_2 = timer_0.__call__(none_type_1)
    var_3 = timer_1.__eq__(timer_1)
    var_0.__getitem__(str_0)
