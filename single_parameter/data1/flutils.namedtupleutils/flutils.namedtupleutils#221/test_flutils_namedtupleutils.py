# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import flutils.namedtupleutils as module_0
import collections as module_1


def test_case_0():
    int_0 = -2449
    module_0.to_namedtuple(int_0)


def test_case_1():
    set_0 = set()
    tuple_0 = (set_0, set_0)
    var_0 = module_0.to_namedtuple(tuple_0)
    str_0 = "A@A"
    list_0 = [str_0, str_0, str_0, str_0]
    var_1 = module_0.to_namedtuple(list_0)
    list_1 = []
    var_2 = module_0.to_namedtuple(list_1)


def test_case_2():
    ordered_dict_0 = module_1.OrderedDict()
    var_0 = module_0.to_namedtuple(ordered_dict_0)


def test_case_3():
    bytes_0 = b""
    module_0.to_namedtuple(bytes_0)


def test_case_4():
    list_0 = []
    var_0 = module_0.to_namedtuple(list_0)


def test_case_5():
    dict_0 = {}
    var_0 = module_0.to_namedtuple(dict_0)


def test_case_6():
    str_0 = "stderr"
    dict_0 = {str_0: str_0}
    ordered_dict_0 = module_1.OrderedDict(**dict_0)
    var_0 = module_0.to_namedtuple(ordered_dict_0)
    var_1 = module_0.to_namedtuple(var_0)


def test_case_7():
    tuple_0 = ()
    var_0 = module_0.to_namedtuple(tuple_0)


def test_case_8():
    ordered_dict_0 = module_1.OrderedDict()
    list_0 = [ordered_dict_0, ordered_dict_0]
    var_0 = module_0.to_namedtuple(list_0)


def test_case_9():
    ordered_dict_0 = module_1.OrderedDict()
    var_0 = module_0.to_namedtuple(ordered_dict_0)
    var_1 = module_0.to_namedtuple(var_0)


def test_case_10():
    str_0 = "LO#W^P,x?giTd\x0c4"
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0, str_0: str_0}
    var_0 = module_0.to_namedtuple(dict_0)
    str_1 = "!R\tQqicZ.ib;xAP&G}/("
    module_0.to_namedtuple(str_1)


def test_case_11():
    float_0 = 132.21
    dict_0 = {float_0: float_0, float_0: float_0}
    tuple_0 = (float_0, dict_0)
    tuple_1 = (dict_0, tuple_0)
    var_0 = module_0.to_namedtuple(tuple_1)
    var_1 = module_0.to_namedtuple(var_0)
    var_2 = module_0.to_namedtuple(var_1)
    var_3 = module_0.to_namedtuple(var_1)


def test_case_12():
    bytes_0 = b"dY\x14a\xdf\x19w/Mu)\xfd\xe8"
    tuple_0 = (bytes_0,)
    float_0 = 132.21300328729265
    dict_0 = {
        tuple_0: float_0,
        float_0: float_0,
        bytes_0: bytes_0,
        float_0: float_0,
        float_0: float_0,
    }
    tuple_1 = (dict_0, tuple_0)
    list_0 = [tuple_1, float_0, tuple_1, tuple_1]
    module_0.to_namedtuple(list_0)


def test_case_13():
    str_0 = "t\t"
    dict_0 = {str_0: str_0}
    var_0 = module_0.to_namedtuple(dict_0)
    var_1 = module_0.to_namedtuple(var_0)
    var_2 = module_0.to_namedtuple(var_1)
    var_3 = module_0.to_namedtuple(var_0)
    bool_0 = False
    var_4 = module_0.to_namedtuple(var_2)
    var_5 = module_0.to_namedtuple(var_0)
    var_6 = module_0.to_namedtuple(var_5)
    module_0.to_namedtuple(bool_0)
