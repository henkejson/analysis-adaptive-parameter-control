# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import flutils.namedtupleutils as module_0
import collections as module_1


def test_case_0():
    int_0 = -1094
    module_0.to_namedtuple(int_0)


def test_case_1():
    tuple_0 = ()
    set_0 = {tuple_0, tuple_0}
    tuple_1 = (set_0, set_0)
    var_0 = module_0.to_namedtuple(tuple_1)
    str_0 = "#3v/ra,~"
    module_0.to_namedtuple(str_0)


def test_case_2():
    str_0 = "rh"
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0}
    var_0 = module_0.to_namedtuple(dict_0)
    var_1 = module_0.to_namedtuple(var_0)


def test_case_3():
    ordered_dict_0 = module_1.OrderedDict()
    var_0 = module_0.to_namedtuple(ordered_dict_0)


def test_case_4():
    str_0 = "4.9\n\x0cc[.k\t"
    module_0.to_namedtuple(str_0)


def test_case_5():
    list_0 = []
    var_0 = module_0.to_namedtuple(list_0)


def test_case_6():
    tuple_0 = ()
    var_0 = module_0.to_namedtuple(tuple_0)
    float_0 = -2700.54
    set_0 = {tuple_0, float_0, float_0}
    dict_0 = {tuple_0: float_0, float_0: set_0}
    var_1 = module_0.to_namedtuple(dict_0)
    var_2 = module_0.to_namedtuple(dict_0)
    none_type_0 = None
    module_0.to_namedtuple(none_type_0)


def test_case_7():
    str_0 = "4.9\n\x0cc[k\t"
    bytes_0 = b'J\xe92^i&\x00`\x89;\xaa\xec"{G\xa0\xfd\xcf\xee;'
    list_0 = [str_0, bytes_0]
    var_0 = module_0.to_namedtuple(list_0)


def test_case_8():
    ordered_dict_0 = module_1.OrderedDict()
    var_0 = module_0.to_namedtuple(ordered_dict_0)
    var_1 = module_0.to_namedtuple(var_0)
    var_2 = module_0.to_namedtuple(var_0)


def test_case_9():
    tuple_0 = ()
    var_0 = module_0.to_namedtuple(tuple_0)


def test_case_10():
    str_0 = "owp<_`e'"
    dict_0 = {str_0: str_0}
    ordered_dict_0 = module_1.OrderedDict(**dict_0)
    var_0 = module_0.to_namedtuple(ordered_dict_0)
    float_0 = 5628.49617
    module_0.to_namedtuple(float_0)


def test_case_11():
    bytes_0 = b"dP{"
    dict_0 = {bytes_0: bytes_0}
    module_0.to_namedtuple(dict_0)


def test_case_12():
    str_0 = "OIL\r"
    dict_0 = {str_0: str_0, str_0: str_0}
    ordered_dict_0 = module_1.OrderedDict(**dict_0)
    var_0 = module_0.to_namedtuple(ordered_dict_0)
    var_1 = module_0.to_namedtuple(ordered_dict_0)
    var_2 = module_0.to_namedtuple(var_1)
    var_3 = module_0.to_namedtuple(var_1)
    var_4 = module_0.to_namedtuple(dict_0)
    var_5 = module_0.to_namedtuple(var_2)
