# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import flutils.namedtupleutils as module_0
import collections as module_1


def test_case_0():
    float_0 = 36.64342
    module_0.to_namedtuple(float_0)


def test_case_1():
    bool_0 = True
    bytes_0 = b"9!\xe9<\xa7\x1a}\xf9\xb1\x8b\x9cA\xba\x1c&-\xc9"
    tuple_0 = (bool_0, bool_0, bytes_0, bool_0)
    var_0 = module_0.to_namedtuple(tuple_0)
    none_type_0 = None
    module_0.to_namedtuple(none_type_0)


def test_case_2():
    ordered_dict_0 = module_1.OrderedDict()
    var_0 = module_0.to_namedtuple(ordered_dict_0)


def test_case_3():
    bytes_0 = b"\x84\xe2\x91c\xe3\xdcq"
    module_0.to_namedtuple(bytes_0)


def test_case_4():
    tuple_0 = ()
    var_0 = module_0.to_namedtuple(tuple_0)


def test_case_5():
    str_0 = "subsequent_indent_len"
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0}
    var_0 = module_0.to_namedtuple(dict_0)


def test_case_6():
    bytes_0 = b"\xc7XWXm;K\xe1:\x13?p\x02p\xcdbMF\xcb"
    list_0 = [bytes_0, bytes_0, bytes_0, bytes_0]
    var_0 = module_0.to_namedtuple(list_0)


def test_case_7():
    str_0 = "subsequent_indent_len"
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0}
    var_0 = module_0.to_namedtuple(dict_0)
    var_1 = module_0.to_namedtuple(var_0)


def test_case_8():
    str_0 = "\x0bmZ~j"
    dict_0 = {str_0: str_0}
    var_0 = module_0.to_namedtuple(dict_0)
    var_1 = module_0.to_namedtuple(var_0)


def test_case_9():
    list_0 = []
    var_0 = module_0.to_namedtuple(list_0)


def test_case_10():
    dict_0 = {}
    var_0 = module_0.to_namedtuple(dict_0)
    var_1 = module_0.to_namedtuple(dict_0)
    var_2 = module_0.to_namedtuple(var_1)
    var_3 = module_0.to_namedtuple(dict_0)
    var_4 = module_0.to_namedtuple(dict_0)
    var_5 = module_0.to_namedtuple(dict_0)
    dict_1 = {var_1: var_4, var_0: var_4, var_5: var_2}
    var_6 = module_0.to_namedtuple(dict_1)
    dict_2 = {}
    var_7 = module_0.to_namedtuple(var_3)
    list_0 = [dict_0, dict_2]
    var_8 = module_0.to_namedtuple(var_3)
    none_type_0 = None
    var_9 = module_0.to_namedtuple(list_0)
    module_0.to_namedtuple(none_type_0)


def test_case_11():
    bytes_0 = b'J9\x11\xd9U\xeb\r\x0e"\n\n\xd2\x08\xc1'
    dict_0 = {bytes_0: bytes_0}
    module_0.to_namedtuple(dict_0)


def test_case_12():
    bool_0 = True
    list_0 = [bool_0, bool_0, bool_0]
    var_0 = module_0.to_namedtuple(list_0)
    str_0 = "k "
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0}
    var_1 = module_0.to_namedtuple(dict_0)
    var_2 = module_0.to_namedtuple(var_1)
    var_3 = module_0.to_namedtuple(var_1)
    var_4 = module_0.to_namedtuple(var_1)
    module_0.to_namedtuple(str_0)
