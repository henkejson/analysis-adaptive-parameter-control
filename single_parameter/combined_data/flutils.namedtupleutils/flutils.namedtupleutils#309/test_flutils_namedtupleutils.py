# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import flutils.namedtupleutils as module_0
import collections as module_1


def test_case_0():
    none_type_0 = None
    module_0.to_namedtuple(none_type_0)


def test_case_1():
    int_0 = 1689
    str_0 = "KCgL"
    dict_0 = {str_0: int_0, int_0: int_0}
    var_0 = module_0.to_namedtuple(dict_0)
    var_1 = module_0.to_namedtuple(var_0)


def test_case_2():
    str_0 = "KCgL"
    dict_0 = {str_0: str_0, str_0: str_0}
    var_0 = module_0.to_namedtuple(dict_0)
    var_1 = module_0.to_namedtuple(var_0)


def test_case_3():
    ordered_dict_0 = module_1.OrderedDict()
    var_0 = module_0.to_namedtuple(ordered_dict_0)


def test_case_4():
    str_0 = "(xl"
    module_0.to_namedtuple(str_0)


def test_case_5():
    list_0 = []
    var_0 = module_0.to_namedtuple(list_0)


def test_case_6():
    dict_0 = {}
    var_0 = module_0.to_namedtuple(dict_0)


def test_case_7():
    complex_0 = 2771.3209 + 4137.1758j
    dict_0 = {complex_0: complex_0, complex_0: complex_0}
    var_0 = module_0.to_namedtuple(dict_0)
    var_1 = module_0.to_namedtuple(var_0)
    var_2 = module_0.to_namedtuple(var_0)
    var_3 = module_0.to_namedtuple(var_0)


def test_case_8():
    tuple_0 = ()
    var_0 = module_0.to_namedtuple(tuple_0)


def test_case_9():
    complex_0 = -2067.24241 - 2459.11j
    set_0 = {complex_0, complex_0, complex_0}
    list_0 = [set_0, complex_0]
    list_1 = [list_0, complex_0]
    var_0 = module_0.to_namedtuple(list_1)
    var_1 = module_0.to_namedtuple(list_0)


def test_case_10():
    int_0 = 1689
    str_0 = "Wraps the single paragraph in the given ``text`` so every line is\n        at most ``width`` characters long. All wrapping options are taken\n        from instance attributes of the\n        :obj:`~flutils.txtutils.AnsiTextWrapper` instance.\n\n        Args:\n            text (str): The text to be wrapped.\n\n        Returns:\n            A ``List[str]`` of output lines, without final newlines.\n            If the wrapped output has no content, the returned list is\n            empty.\n        "
    int_1 = 365
    str_1 = "KCgL"
    dict_0 = {int_1: int_1, str_1: int_0, str_0: int_0, int_1: int_1}
    list_0 = [dict_0, str_1, str_1]
    tuple_0 = (int_1, int_1, list_0, str_1)
    tuple_1 = (int_0, str_0, tuple_0)
    var_0 = module_0.to_namedtuple(tuple_1)
    var_1 = module_0.to_namedtuple(var_0)
    bool_0 = False
    module_0.to_namedtuple(bool_0)


def test_case_11():
    tuple_0 = ()
    var_0 = module_0.to_namedtuple(tuple_0)
    bytes_0 = b"?;\x00s\xee/\xbc_\x85P\n\x1b\xcf"
    var_1 = module_0.to_namedtuple(tuple_0)
    complex_0 = -2443.67447 + 305.675611j
    var_2 = module_0.to_namedtuple(tuple_0)
    var_3 = module_0.to_namedtuple(var_0)
    dict_0 = {bytes_0: bytes_0, complex_0: bytes_0, bytes_0: bytes_0, bytes_0: bytes_0}
    tuple_1 = (dict_0,)
    module_0.to_namedtuple(tuple_1)


def test_case_12():
    bool_0 = False
    str_0 = "\rNH6k"
    str_1 = "Qx6XofbyA|9.8K2\\"
    dict_0 = {str_0: bool_0, str_1: str_1}
    ordered_dict_0 = module_1.OrderedDict(**dict_0)
    tuple_0 = (bool_0, ordered_dict_0)
    var_0 = module_0.to_namedtuple(tuple_0)
    bytes_0 = b"\xcd\xce\x18!B\xb1`3\x0c\xe1\xb0$\x8d0\x92\xf8\xc0\xca"
    module_0.to_namedtuple(bytes_0)
