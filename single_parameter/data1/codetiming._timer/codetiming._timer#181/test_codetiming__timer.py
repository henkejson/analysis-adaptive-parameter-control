# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import codetiming._timer as module_0
import dataclasses as module_1


def test_case_0():
    float_arg_0 = module_0.FloatArg()


def test_case_1():
    timer_0 = module_0.Timer()
    none_type_0 = timer_0.start()
    timer_0.__enter__()


def test_case_2():
    timer_0 = module_0.Timer()
    list_0 = [timer_0, timer_0, timer_0, timer_0]
    none_type_0 = timer_0.start()
    none_type_1 = timer_0.__exit__(*list_0)


def test_case_3():
    str_0 = "u\n]bX2u\r"
    timer_0 = module_0.Timer(str_0)
    timer_0.__exit__()


def test_case_4():
    str_0 = "?X|U6p h"
    timer_0 = module_0.Timer(text=str_0, initial_text=str_0)
    timer_1 = timer_0.__enter__()
    float_0 = timer_0.stop()


def test_case_5():
    bool_0 = True
    none_type_0 = None
    timer_0 = module_0.Timer(text=bool_0, logger=none_type_0)
    none_type_1 = timer_0.start()
    float_arg_0 = module_0.FloatArg()
    timer_0.__enter__()


def test_case_6():
    str_0 = ">{\tNF*aw`\x0c"
    timer_0 = module_0.Timer(str_0)
    var_0 = timer_0.__call__(str_0)
    none_type_0 = timer_0.start()
    none_type_1 = timer_0.__exit__()
    var_1 = timer_0.__eq__(var_0)
    var_1.__enter__()


def test_case_7():
    bool_0 = True
    none_type_0 = None
    timer_0 = module_0.Timer(text=bool_0, logger=none_type_0)
    var_0 = timer_0.__repr__()
    none_type_1 = timer_0.start()
    float_0 = timer_0.stop()
    timer_1 = timer_0.__enter__()
    var_1 = timer_0.__eq__(none_type_0)
    module_0.TimerError(*var_1)


def test_case_8():
    float_0 = 2239.2173
    timer_0 = module_0.Timer(initial_text=float_0, logger=float_0)
    timer_0.__enter__()


def test_case_9():
    str_0 = "?X|U6p h"
    timer_0 = module_0.Timer(text=str_0, initial_text=str_0)
    timer_1 = timer_0.__enter__()
    float_0 = timer_0.stop()
    none_type_0 = None
    var_0 = timer_1.__eq__(none_type_0)
    timer_2 = module_0.Timer(var_0, float_0, var_0)
    timer_3 = timer_2.__enter__()
    var_1 = timer_2.__eq__(timer_2)
    timer_1.stop()


def test_case_10():
    var_0 = module_1.dataclass()
    timer_0 = module_0.Timer(text=var_0, initial_text=var_0)
    none_type_0 = timer_0.start()
    timer_0.__exit__()
