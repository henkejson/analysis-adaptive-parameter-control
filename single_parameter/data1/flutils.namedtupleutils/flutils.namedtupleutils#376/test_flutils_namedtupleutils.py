# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import flutils.namedtupleutils as module_0
import collections as module_1


def test_case_0():
    bool_0 = True
    module_0.to_namedtuple(bool_0)


def test_case_1():
    float_0 = 1515.90462
    list_0 = [float_0]
    var_0 = module_0.to_namedtuple(list_0)
    ordered_dict_0 = module_1.OrderedDict()
    var_1 = module_0.to_namedtuple(ordered_dict_0)


def test_case_2():
    str_0 = "latin1"
    dict_0 = {str_0: str_0, str_0: str_0}
    var_0 = module_0.to_namedtuple(dict_0)
    var_1 = module_0.to_namedtuple(var_0)


def test_case_3():
    ordered_dict_0 = module_1.OrderedDict()
    var_0 = module_0.to_namedtuple(ordered_dict_0)


def test_case_4():
    str_0 = "fY$"
    module_0.to_namedtuple(str_0)


def test_case_5():
    list_0 = []
    var_0 = module_0.to_namedtuple(list_0)


def test_case_6():
    bool_0 = False
    str_0 = "[WlJ.Bi^Kc.TH"
    dict_0 = {bool_0: bool_0, bool_0: bool_0, str_0: str_0, str_0: bool_0}
    tuple_0 = (dict_0, str_0)
    var_0 = module_0.to_namedtuple(tuple_0)


def test_case_7():
    ordered_dict_0 = module_1.OrderedDict()
    list_0 = [ordered_dict_0, ordered_dict_0, ordered_dict_0]
    var_0 = module_0.to_namedtuple(list_0)
    var_1 = module_0.to_namedtuple(var_0)


def test_case_8():
    tuple_0 = ()
    var_0 = module_0.to_namedtuple(tuple_0)


def test_case_9():
    ordered_dict_0 = module_1.OrderedDict()
    list_0 = [ordered_dict_0, ordered_dict_0, ordered_dict_0]
    var_0 = module_0.to_namedtuple(list_0)


def test_case_10():
    complex_0 = -501 - 900j
    bytes_0 = b"/\xea{\xbf\xcf1"
    dict_0 = {
        complex_0: complex_0,
        complex_0: complex_0,
        bytes_0: bytes_0,
        bytes_0: bytes_0,
    }
    module_0.to_namedtuple(dict_0)


def test_case_11():
    str_0 = "j\n"
    dict_0 = {str_0: str_0, str_0: str_0}
    ordered_dict_0 = module_1.OrderedDict(**dict_0)
    float_0 = -932.862889
    list_0 = [ordered_dict_0, ordered_dict_0, float_0, float_0]
    var_0 = module_0.to_namedtuple(list_0)
    var_1 = module_0.to_namedtuple(list_0)
    var_2 = module_0.to_namedtuple(var_1)
    module_1.namedtuple(var_1, var_0, rename=var_1)
