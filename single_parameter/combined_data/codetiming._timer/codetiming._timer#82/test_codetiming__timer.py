# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import codetiming._timer as module_0
import dataclasses as module_1
import codetiming._timers as module_2


def test_case_0():
    float_arg_0 = module_0.FloatArg()


def test_case_1():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    timer_0.start()


def test_case_2():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    none_type_0 = timer_1.__exit__()


def test_case_3():
    timer_0 = module_0.Timer()
    none_type_0 = timer_0.__eq__(timer_0)
    timer_0.stop()


def test_case_4():
    bytes_0 = b"\x02\xd3Y\x1b\xdf\xbf"
    float_0 = 43.559323141498396
    timer_0 = module_0.Timer(bytes_0, initial_text=bytes_0, logger=float_0)
    timer_1 = module_0.Timer()
    timer_2 = timer_1.__enter__()
    none_type_0 = None
    timer_3 = module_0.Timer(initial_text=none_type_0)
    var_0 = timer_2.__eq__(float_0)
    none_type_1 = timer_2.__exit__()
    timer_0.__enter__()


def test_case_5():
    str_0 = "&aLX`mZ+\rJ\"'"
    timer_0 = module_0.Timer(str_0)
    timer_1 = timer_0.__enter__()
    var_0 = module_1.dataclass()
    var_1 = timer_0.__repr__()
    none_type_0 = timer_0.__exit__(*var_1)
    var_2 = module_1.dataclass(repr=str_0, eq=timer_1)
    timer_error_0 = module_0.TimerError(*var_1)
    var_3 = var_2.__repr__()
    bool_0 = False
    list_0 = [bool_0, bool_0]
    module_2.Timers(*list_0)


def test_case_6():
    bool_0 = True
    timer_0 = module_0.Timer(initial_text=bool_0, logger=bool_0)
    timer_0.__enter__()


def test_case_7():
    timer_error_0 = module_0.TimerError()
    none_type_0 = None
    str_0 = "*R:9!Wa"
    timer_0 = module_0.Timer(none_type_0, initial_text=str_0)
    timer_1 = timer_0.__enter__()
    module_0.FloatArg(**none_type_0)


def test_case_8():
    bytes_0 = b"\x02\xd3Y\x1b\xdf\xbf"
    float_0 = 43.559323141498396
    none_type_0 = None
    timer_0 = module_0.Timer(bytes_0, initial_text=bytes_0, logger=none_type_0)
    var_0 = timer_0.__repr__()
    timer_1 = timer_0.__enter__()
    var_1 = timer_1.__eq__(float_0)
    none_type_1 = timer_1.__exit__()
    var_2 = timer_1.__eq__(bytes_0)
    timer_2 = timer_0.__enter__()
    timer_2.copy()
