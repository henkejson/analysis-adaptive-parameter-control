# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import flutils.namedtupleutils as module_0
import collections as module_1


def test_case_0():
    int_0 = 730
    module_0.to_namedtuple(int_0)


def test_case_1():
    bool_0 = False
    dict_0 = {bool_0: bool_0, bool_0: bool_0}
    list_0 = [dict_0, dict_0, bool_0]
    var_0 = module_0.to_namedtuple(list_0)
    var_1 = module_0.to_namedtuple(list_0)
    module_0.to_namedtuple(bool_0)


def test_case_2():
    str_0 = "camel_to_underscore"
    dict_0 = {str_0: str_0, str_0: str_0}
    var_0 = module_0.to_namedtuple(dict_0)
    var_1 = module_0.to_namedtuple(var_0)


def test_case_3():
    dict_0 = {}
    var_0 = module_0.to_namedtuple(dict_0)


def test_case_4():
    bytes_0 = b"\xf8\xb6\x08\x02\xbd\xce\xe3\xf4\xeb\x97\x81<\x04\x12.\xbc0r\x04\xb1"
    module_0.to_namedtuple(bytes_0)


def test_case_5():
    tuple_0 = ()
    var_0 = module_0.to_namedtuple(tuple_0)


def test_case_6():
    bool_0 = False
    dict_0 = {bool_0: bool_0, bool_0: bool_0}
    list_0 = [dict_0]
    var_0 = module_0.to_namedtuple(list_0)
    int_0 = -1232
    set_0 = {int_0, int_0, int_0, int_0}
    module_0.to_namedtuple(set_0)


def test_case_7():
    dict_0 = {}
    var_0 = module_0.to_namedtuple(dict_0)
    var_1 = module_0.to_namedtuple(var_0)


def test_case_8():
    list_0 = []
    var_0 = module_0.to_namedtuple(list_0)


def test_case_9():
    str_0 = "0\x0bDu-hV}\tov*G"
    tuple_0 = (str_0,)
    var_0 = module_0.to_namedtuple(tuple_0)
    bool_0 = True
    module_0.to_namedtuple(bool_0)


def test_case_10():
    ordered_dict_0 = module_1.OrderedDict()
    var_0 = module_0.to_namedtuple(ordered_dict_0)


def test_case_11():
    bytes_0 = b"\xae\xff@?"
    dict_0 = {bytes_0: bytes_0, bytes_0: bytes_0}
    tuple_0 = (bytes_0, dict_0)
    module_0.to_namedtuple(tuple_0)


def test_case_12():
    str_0 = "dvs#ssIz^>G"
    dict_0 = {str_0: str_0}
    var_0 = module_0.to_namedtuple(dict_0)
    str_1 = "j_"
    module_0.to_namedtuple(str_1)


def test_case_13():
    str_0 = "\nO"
    dict_0 = {str_0: str_0, str_0: str_0}
    var_0 = module_0.to_namedtuple(dict_0)
    var_1 = module_0.to_namedtuple(dict_0)
    var_2 = module_0.to_namedtuple(var_1)
    var_3 = module_0.to_namedtuple(var_1)
    var_4 = module_0.to_namedtuple(var_1)
    module_0.to_namedtuple(str_0)
