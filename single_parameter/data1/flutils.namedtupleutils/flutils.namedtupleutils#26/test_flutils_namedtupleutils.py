# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import flutils.namedtupleutils as module_0
import collections as module_1


def test_case_0():
    none_type_0 = None
    module_0.to_namedtuple(none_type_0)


def test_case_1():
    int_0 = -289
    tuple_0 = (int_0, int_0)
    var_0 = module_0.to_namedtuple(tuple_0)


def test_case_2():
    str_0 = "N7"
    str_1 = "0fl"
    str_2 = "\x0cvNnk'7JYJ=-ehBCiFQ"
    dict_0 = {str_0: str_1, str_1: str_1, str_2: str_1}
    ordered_dict_0 = module_1.OrderedDict(**dict_0)
    var_0 = module_0.to_namedtuple(dict_0)


def test_case_3():
    dict_0 = {}
    var_0 = module_0.to_namedtuple(dict_0)


def test_case_4():
    str_0 = "Dt'SF@m- Ri;2EA4\x0cOm"
    module_0.to_namedtuple(str_0)


def test_case_5():
    list_0 = []
    var_0 = module_0.to_namedtuple(list_0)


def test_case_6():
    tuple_0 = ()
    str_0 = "N7"
    dict_0 = {str_0: tuple_0, tuple_0: tuple_0, tuple_0: tuple_0}
    ordered_dict_0 = module_1.OrderedDict(**dict_0)
    var_0 = module_0.to_namedtuple(ordered_dict_0)
    var_1 = module_0.to_namedtuple(var_0)


def test_case_7():
    ordered_dict_0 = module_1.OrderedDict()
    var_0 = module_0.to_namedtuple(ordered_dict_0)


def test_case_8():
    tuple_0 = ()
    str_0 = "N7"
    dict_0 = {str_0: tuple_0, str_0: tuple_0, str_0: str_0}
    ordered_dict_0 = module_1.OrderedDict(**dict_0)
    var_0 = module_0.to_namedtuple(ordered_dict_0)
    var_1 = module_0.to_namedtuple(var_0)


def test_case_9():
    tuple_0 = ()
    bytes_0 = b"h+\xd8\x97\xf1C\x87\xfea\xeb\x7f2\x91"
    dict_0 = {bytes_0: tuple_0}
    module_0.to_namedtuple(dict_0)


def test_case_10():
    tuple_0 = ()
    str_0 = "0fl"
    str_1 = "\x0cvDnk!7JYJ=-ehBCiFQ"
    dict_0 = {str_0: tuple_0, str_0: tuple_0, str_1: str_0}
    ordered_dict_0 = module_1.OrderedDict(**dict_0)
    list_0 = [tuple_0, ordered_dict_0]
    var_0 = module_0.to_namedtuple(list_0)
    var_1 = module_0.to_namedtuple(var_0)


def test_case_11():
    tuple_0 = ()
    var_0 = module_0.to_namedtuple(tuple_0)


def test_case_12():
    tuple_0 = ()
    str_0 = "N\x0b"
    str_1 = "0fl"
    str_2 = "\x0cvDnk!7JYJ=-ehBCiFQ"
    dict_0 = {str_0: tuple_0, str_1: tuple_0, str_2: str_1}
    ordered_dict_0 = module_1.OrderedDict(**dict_0)
    list_0 = [tuple_0, ordered_dict_0]
    var_0 = module_0.to_namedtuple(list_0)
    var_1 = module_0.to_namedtuple(list_0)
    module_1.namedtuple(str_1, str_2, module=dict_0)
