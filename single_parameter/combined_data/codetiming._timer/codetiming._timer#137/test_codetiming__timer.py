# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import codetiming._timer as module_0
import collections as module_1
import codetiming._timers as module_2
import builtins as module_3
import contextlib as module_4


def test_case_0():
    float_arg_0 = module_0.FloatArg()


def test_case_1():
    timer_0 = module_0.Timer()
    timer_1 = module_0.Timer(initial_text=timer_0)
    timer_2 = timer_1.__enter__()


def test_case_2():
    bool_0 = True
    timer_0 = module_0.Timer(bool_0)
    timer_0.stop()


def test_case_3():
    timer_0 = module_0.Timer()
    none_type_0 = timer_0.start()
    float_0 = timer_0.stop()
    timer_error_0 = module_0.TimerError()


def test_case_4():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    none_type_0 = timer_1.__exit__()
    timer_2 = module_0.Timer(initial_text=timer_1)
    timer_3 = timer_2.__enter__()
    timer_3.start()


def test_case_5():
    bool_0 = True
    timer_0 = module_0.Timer(bool_0)
    none_type_0 = timer_0.start()
    float_0 = timer_0.stop()
    timer_1 = module_0.Timer()
    user_dict_0 = module_1.UserDict()
    bool_1 = True
    var_0 = user_dict_0.__ror__(bool_1)
    var_1 = var_0.__repr__()
    var_0.stop()


def test_case_6():
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    timer_2 = module_0.Timer(initial_text=timer_1)
    timer_3 = timer_2.__enter__()


def test_case_7():
    timer_0 = module_0.Timer()
    timer_1 = module_0.Timer(initial_text=timer_0)
    timer_2 = timer_1.__enter__()
    timer_2.start()


def test_case_8():
    str_0 = "P\n44<Q-dc"
    none_type_0 = None
    timer_0 = module_0.Timer(initial_text=str_0, logger=none_type_0)
    none_type_1 = timer_0.start()
    var_0 = timer_0.__call__(none_type_0)
    var_0.clear()


def test_case_9():
    str_0 = "P\n44<Q-dc"
    none_type_0 = None
    timer_0 = module_0.Timer(initial_text=str_0, logger=none_type_0)
    none_type_1 = timer_0.start()
    float_0 = timer_0.stop()
    var_0 = timer_0.__call__(none_type_0)
    var_0.clear()


def test_case_10():
    str_0 = "T"
    timer_0 = module_0.Timer(str_0, initial_text=str_0)
    none_type_0 = timer_0.start()
    dict_0 = {}
    module_1.UserDict(str_0, **dict_0)


def test_case_11():
    timer_0 = module_0.Timer()
    none_type_0 = timer_0.start()
    timer_1 = module_0.Timer(none_type_0, timer_0)
    none_type_1 = timer_1.start()
    float_0 = timer_1.stop()
    timers_0 = module_2.Timers()
    var_0 = timers_0.copy()
    var_1 = var_0.__iter__()
    var_2 = var_1.__iter__()
    var_2.__delitem__(none_type_1)


def test_case_12():
    object_0 = module_3.object()
    list_0 = [object_0, object_0]
    timer_0 = module_0.Timer(list_0, initial_text=list_0)
    none_type_0 = None
    context_decorator_0 = module_4.ContextDecorator()
    var_0 = context_decorator_0.__call__(none_type_0)
    timer_1 = timer_0.__enter__()
    timer_1.__exit__()
