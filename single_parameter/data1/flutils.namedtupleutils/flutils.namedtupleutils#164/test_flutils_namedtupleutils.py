# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import flutils.namedtupleutils as module_0
import collections as module_1


def test_case_0():
    bool_0 = False
    module_0.to_namedtuple(bool_0)


def test_case_1():
    float_0 = 40.15171
    list_0 = [float_0, float_0]
    var_0 = module_0.to_namedtuple(list_0)


def test_case_2():
    ordered_dict_0 = module_1.OrderedDict()
    var_0 = module_0.to_namedtuple(ordered_dict_0)


def test_case_3():
    str_0 = "0^Da[O*r0ct"
    module_0.to_namedtuple(str_0)


def test_case_4():
    list_0 = []
    var_0 = module_0.to_namedtuple(list_0)


def test_case_5():
    str_0 = "ie9USc9"
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0, str_0: str_0}
    var_0 = module_0.to_namedtuple(dict_0)


def test_case_6():
    tuple_0 = ()
    var_0 = module_0.to_namedtuple(tuple_0)


def test_case_7():
    str_0 = "ie9USc9"
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0, str_0: str_0}
    var_0 = module_0.to_namedtuple(dict_0)
    var_1 = module_0.to_namedtuple(var_0)


def test_case_8():
    ordered_dict_0 = module_1.OrderedDict()
    var_0 = module_0.to_namedtuple(ordered_dict_0)
    var_1 = module_0.to_namedtuple(var_0)


def test_case_9():
    bytes_0 = b"\xcdtv\xd0\xd7\x06u\xce\xcaH\xff\xef6\x13:\x9c"
    bool_0 = True
    ordered_dict_0 = module_1.OrderedDict()
    var_0 = module_0.to_namedtuple(ordered_dict_0)
    tuple_0 = (bytes_0, bool_0, var_0, var_0)
    var_1 = module_0.to_namedtuple(tuple_0)
    var_2 = module_0.to_namedtuple(var_1)
    dict_0 = {var_1: var_0, bytes_0: var_1, var_0: ordered_dict_0, var_0: var_0}
    list_0 = [var_1, var_0, dict_0]
    module_0.to_namedtuple(list_0)


def test_case_10():
    str_0 = ""
    str_1 = "ie9USc9"
    dict_0 = {str_0: str_0, str_1: str_1, str_0: str_0, str_1: str_0}
    var_0 = module_0.to_namedtuple(dict_0)
    var_1 = module_0.to_namedtuple(var_0)


def test_case_11():
    str_0 = "n<[^3L44(~H^DJKD`g\tf"
    str_1 = "oi9USc "
    dict_0 = {
        str_0: str_0,
        str_1: str_1,
        str_1: str_1,
        str_0: str_0,
        str_0: str_0,
        str_1: str_0,
    }
    ordered_dict_0 = module_1.OrderedDict(**dict_0)
    var_0 = module_0.to_namedtuple(ordered_dict_0)
    var_1 = module_0.to_namedtuple(var_0)
    var_2 = module_0.to_namedtuple(var_1)
    var_3 = module_0.to_namedtuple(var_1)
    module_1.OrderedDict(**var_1)
