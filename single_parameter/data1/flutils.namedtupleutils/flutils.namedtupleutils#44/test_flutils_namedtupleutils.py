# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import flutils.namedtupleutils as module_0
import collections as module_1


def test_case_0():
    bool_0 = False
    module_0.to_namedtuple(bool_0)


def test_case_1():
    dict_0 = {}
    set_0 = set()
    list_0 = [dict_0, dict_0, set_0]
    var_0 = module_0.to_namedtuple(list_0)


def test_case_2():
    str_0 = "VTp\n"
    dict_0 = {str_0: str_0}
    ordered_dict_0 = module_1.OrderedDict(**dict_0)
    list_0 = [ordered_dict_0, dict_0]
    int_0 = 629
    tuple_0 = (list_0, int_0, int_0)
    var_0 = module_0.to_namedtuple(tuple_0)


def test_case_3():
    ordered_dict_0 = module_1.OrderedDict()
    list_0 = [ordered_dict_0, ordered_dict_0]
    var_0 = module_0.to_namedtuple(list_0)


def test_case_4():
    str_0 = "%"
    module_0.to_namedtuple(str_0)


def test_case_5():
    tuple_0 = ()
    var_0 = module_0.to_namedtuple(tuple_0)


def test_case_6():
    bool_0 = True
    dict_0 = {bool_0: bool_0}
    var_0 = module_0.to_namedtuple(dict_0)
    module_0.to_namedtuple(bool_0)


def test_case_7():
    dict_0 = {}
    set_0 = module_0.to_namedtuple(dict_0)


def test_case_8():
    bytes_0 = b"[\xba\x87\x87\xcf|\x1fNr\xba\xb8s\x7fcQG\xb0)L\xf2"
    list_0 = [bytes_0, bytes_0, bytes_0]
    var_0 = module_0.to_namedtuple(list_0)


def test_case_9():
    ordered_dict_0 = module_1.OrderedDict()
    var_0 = module_0.to_namedtuple(ordered_dict_0)
    var_1 = module_0.to_namedtuple(var_0)


def test_case_10():
    bytes_0 = b"n?\xad\x11"
    bytes_1 = b""
    tuple_0 = ()
    var_0 = module_0.to_namedtuple(tuple_0)
    tuple_1 = (bytes_0, bytes_1, var_0)
    set_0 = set()
    dict_0 = {bytes_0: set_0}
    list_0 = [tuple_1, tuple_0, dict_0, tuple_0]
    module_0.to_namedtuple(list_0)


def test_case_11():
    ordered_dict_0 = module_1.OrderedDict()
    var_0 = module_0.to_namedtuple(ordered_dict_0)
    var_1 = module_0.to_namedtuple(var_0)
    var_2 = module_0.to_namedtuple(var_0)
    var_3 = module_0.to_namedtuple(var_2)
    str_0 = "decode"
    str_1 = "Ln+)|l4:BNGP{;"
    dict_0 = {str_0: str_0, str_0: ordered_dict_0, str_1: var_0}
    ordered_dict_1 = module_1.OrderedDict(**dict_0)
    var_4 = module_0.to_namedtuple(ordered_dict_1)
    var_5 = module_0.to_namedtuple(var_4)


def test_case_12():
    ordered_dict_0 = module_1.OrderedDict()
    var_0 = module_0.to_namedtuple(ordered_dict_0)
    var_1 = module_0.to_namedtuple(ordered_dict_0)
    var_2 = module_0.to_namedtuple(var_1)
    str_0 = "decode"
    dict_0 = {str_0: str_0, str_0: ordered_dict_0, var_0: var_0}
    ordered_dict_1 = module_1.OrderedDict(**dict_0)
    var_3 = module_0.to_namedtuple(ordered_dict_1)
    var_4 = module_0.to_namedtuple(var_3)
