# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import flutils.namedtupleutils as module_0
import collections as module_1
import builtins as module_2


def test_case_0():
    none_type_0 = None
    module_0.to_namedtuple(none_type_0)


def test_case_1():
    float_0 = 1373.18
    bool_0 = False
    tuple_0 = (bool_0,)
    tuple_1 = (float_0, tuple_0, bool_0)
    var_0 = module_0.to_namedtuple(tuple_1)


def test_case_2():
    str_0 = "a"
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0, str_0: str_0}
    ordered_dict_0 = module_1.OrderedDict(**dict_0)
    var_0 = module_0.to_namedtuple(ordered_dict_0)


def test_case_3():
    ordered_dict_0 = module_1.OrderedDict()
    var_0 = module_0.to_namedtuple(ordered_dict_0)


def test_case_4():
    str_0 = "7Im "
    module_0.to_namedtuple(str_0)


def test_case_5():
    tuple_0 = ()
    var_0 = module_0.to_namedtuple(tuple_0)


def test_case_6():
    float_0 = 308.28
    dict_0 = {float_0: float_0}
    var_0 = module_0.to_namedtuple(dict_0)
    ordered_dict_0 = module_1.OrderedDict()
    var_1 = module_0.to_namedtuple(ordered_dict_0)


def test_case_7():
    str_0 = "a"
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0, str_0: str_0}
    ordered_dict_0 = module_0.to_namedtuple(dict_0)
    var_0 = module_0.to_namedtuple(ordered_dict_0)


def test_case_8():
    ordered_dict_0 = module_1.OrderedDict()
    var_0 = module_0.to_namedtuple(ordered_dict_0)
    var_1 = module_0.to_namedtuple(var_0)


def test_case_9():
    list_0 = []
    var_0 = module_0.to_namedtuple(list_0)


def test_case_10():
    bytes_0 = b"kB\xf82\xa27\xcfJ\x01b-\x05\xbdQXv\x1e\x08"
    list_0 = [bytes_0, bytes_0]
    var_0 = module_0.to_namedtuple(list_0)
    float_0 = 3235.04
    module_0.to_namedtuple(float_0)


def test_case_11():
    bool_0 = True
    bytes_0 = b"\x96\xc2\xe8\xe6.\xbe\x91\x1a\xackt\xcd0g\x19\xfe\xff\xb4"
    str_0 = "\x0cO?ni,07K?SC*N"
    dict_0 = {bool_0: bool_0, bool_0: str_0, bytes_0: bytes_0}
    module_0.to_namedtuple(dict_0)


def test_case_12():
    str_0 = "a"
    str_1 = "E\rD1n||3Qz(|n_"
    dict_0 = {str_0: str_0, str_1: str_1, str_0: str_1, str_1: str_1}
    ordered_dict_0 = module_1.OrderedDict(**dict_0)
    var_0 = module_0.to_namedtuple(ordered_dict_0)


def test_case_13():
    str_0 = "Unable to run the command "
    str_1 = "\x0birectory_present"
    dict_0 = {str_0: str_0, str_0: str_0, str_1: str_0}
    ordered_dict_0 = module_1.OrderedDict(**dict_0)
    int_0 = 279
    list_0 = [ordered_dict_0, int_0, dict_0, str_1]
    var_0 = module_0.to_namedtuple(list_0)
    var_1 = module_0.to_namedtuple(var_0)
    int_1 = -136
    dict_1 = {int_1: int_1}
    var_2 = module_0.to_namedtuple(dict_1)
    var_3 = module_0.to_namedtuple(var_1)
    object_0 = module_2.object()
    var_4 = module_0.to_namedtuple(var_3)
    module_0.to_namedtuple(int_1)
