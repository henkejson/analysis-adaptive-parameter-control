# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import codetiming._timer as module_0
import collections as module_1


def test_case_0():
    float_arg_0 = module_0.FloatArg()


def test_case_1():
    bool_0 = True
    timer_0 = module_0.Timer(initial_text=bool_0)
    none_type_0 = timer_0.start()


def test_case_2():
    complex_0 = 253 + 999.7738j
    timer_0 = module_0.Timer(complex_0)
    var_0 = timer_0.__repr__()
    timer_0.stop()


def test_case_3():
    dict_0 = {}
    timer_0 = module_0.Timer()
    bool_0 = True
    timer_1 = module_0.Timer(initial_text=bool_0)
    float_arg_0 = module_0.FloatArg(*dict_0)
    timer_0.__exit__()


def test_case_4():
    dict_0 = {}
    float_arg_0 = module_0.FloatArg()
    timer_0 = module_0.Timer()
    timer_error_0 = module_0.TimerError()
    timer_1 = timer_0.__enter__()
    float_arg_1 = module_0.FloatArg(*dict_0)


def test_case_5():
    none_type_0 = None
    timer_0 = module_0.Timer(none_type_0, logger=none_type_0)
    timer_1 = timer_0.__enter__()
    float_0 = timer_1.stop()


def test_case_6():
    dict_0 = {}
    float_arg_0 = module_0.FloatArg()
    timer_0 = module_0.Timer()
    timer_1 = timer_0.__enter__()
    float_arg_1 = module_0.FloatArg(*dict_0)
    none_type_0 = timer_0.__exit__()


def test_case_7():
    float_arg_0 = module_0.FloatArg()
    timer_0 = module_0.Timer()
    bool_0 = True
    timer_1 = module_0.Timer(initial_text=bool_0)
    user_dict_0 = module_1.UserDict()
    timer_2 = timer_1.__enter__()
    timer_2.start()


def test_case_8():
    bytes_0 = b";\x04\xe0\x8e\x95\xc51%\x8f\xbf\xe9dh\xedt\xb6\x10\xa6\x12"
    none_type_0 = None
    timer_0 = module_0.Timer(bytes_0, initial_text=none_type_0)
    var_0 = timer_0.__repr__()
    timer_1 = timer_0.__enter__()
    none_type_1 = None
    none_type_2 = timer_1.__exit__()
    var_1 = timer_1.__eq__(none_type_1)
    bool_0 = True
    timer_2 = module_0.Timer(logger=bool_0)
    timer_3 = timer_2.__enter__()
    timer_4 = module_0.Timer(text=var_0)
    float_arg_0 = module_0.FloatArg()
    bytes_1 = b"Bl\xd2o\xd7\x13+"
    var_1.__ror__(bytes_1)


def test_case_9():
    str_0 = '$$gn"5.UpcZ_GNB#'
    timer_0 = module_0.Timer(str_0, initial_text=str_0)
    var_0 = timer_0.__repr__()
    timer_1 = timer_0.__enter__()
    none_type_0 = timer_1.__exit__()
    var_1 = timer_0.__eq__(var_0)
    timer_2 = module_0.Timer()
    timer_3 = module_0.Timer()
    timer_4 = timer_3.__enter__()
    timer_1.stop()


def test_case_10():
    bytes_0 = b";\x04\xe0\x8e\x95\xc51%\x8f\xbf\xe9dh\xedt\xb6\x10\xa6\x12"
    timer_0 = module_0.Timer(initial_text=bytes_0)
    var_0 = timer_0.__repr__()
    timer_1 = timer_0.__enter__()
    none_type_0 = None
    timer_2 = module_0.Timer()
    var_1 = timer_1.__eq__(var_0)
    timer_3 = module_0.Timer(timer_1, logger=none_type_0)
    timer_4 = module_0.Timer(text=timer_0)
    timer_5 = timer_4.__enter__()
    float_0 = timer_4.stop()
    bytes_1 = b""
    var_1.__delitem__(bytes_1)


def test_case_11():
    bytes_0 = b";\x04\xe0\x8e\x95\xc51%\x8f\xbf\xe9dh\xedt\xb6\x10\xa6\x12"
    bytes_1 = b";\x04\xe0\x8e\x95\xc51%\x8f\xbf\xe9dht\xb6\x10\xa6\x12"
    timer_0 = module_0.Timer(bytes_0, initial_text=bytes_1, logger=bytes_1)
    var_0 = timer_0.__repr__()
    timer_0.__enter__()
