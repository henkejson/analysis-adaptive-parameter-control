# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import flutils.namedtupleutils as module_0
import collections as module_1


def test_case_0():
    bool_0 = True
    module_0.to_namedtuple(bool_0)


def test_case_1():
    bool_0 = False
    list_0 = [bool_0, bool_0, bool_0, bool_0]
    var_0 = module_0.to_namedtuple(list_0)


def test_case_2():
    tuple_0 = ()
    tuple_1 = (tuple_0,)
    dict_0 = {tuple_1: tuple_1, tuple_1: tuple_1, tuple_1: tuple_1, tuple_1: tuple_1}
    var_0 = module_0.to_namedtuple(dict_0)
    var_1 = module_0.to_namedtuple(var_0)


def test_case_3():
    ordered_dict_0 = module_1.OrderedDict()
    var_0 = module_0.to_namedtuple(ordered_dict_0)


def test_case_4():
    str_0 = "LoX-&Y,scpTB]BGHEuz"
    module_0.to_namedtuple(str_0)


def test_case_5():
    list_0 = []
    var_0 = module_0.to_namedtuple(list_0)


def test_case_6():
    str_0 = "as_escaped_utf8_literal"
    dict_0 = {str_0: str_0}
    list_0 = [str_0, str_0, dict_0]
    list_1 = [list_0, list_0]
    var_0 = module_0.to_namedtuple(list_1)


def test_case_7():
    dict_0 = {}
    var_0 = module_0.to_namedtuple(dict_0)


def test_case_8():
    bytes_0 = b"\xaeC\xef"
    list_0 = [bytes_0, bytes_0]
    var_0 = module_0.to_namedtuple(list_0)
    var_1 = module_0.to_namedtuple(var_0)


def test_case_9():
    str_0 = "$VVxks4XoyDX\n|,"
    none_type_0 = None
    dict_0 = {str_0: none_type_0, str_0: none_type_0, str_0: none_type_0}
    bool_0 = False
    list_0 = [bool_0, bool_0, bool_0, str_0]
    var_0 = module_0.to_namedtuple(list_0)
    dict_1 = {bool_0: bool_0, bool_0: bool_0, bool_0: bool_0, bool_0: bool_0}
    var_1 = module_0.to_namedtuple(dict_0)
    var_2 = module_0.to_namedtuple(dict_1)


def test_case_10():
    tuple_0 = ()
    var_0 = module_0.to_namedtuple(tuple_0)


def test_case_11():
    bytes_0 = b"QaW&=2"
    bool_0 = True
    dict_0 = {bytes_0: bool_0}
    module_0.to_namedtuple(dict_0)


def test_case_12():
    str_0 = "as_escaped_utf8_literal"
    dict_0 = {str_0: str_0}
    list_0 = [str_0, str_0, dict_0]
    list_1 = module_0.to_namedtuple(list_0)
    var_0 = module_0.to_namedtuple(list_1)


def test_case_13():
    str_0 = "addtl_dttr "
    none_type_0 = None
    dict_0 = {str_0: none_type_0, str_0: none_type_0, str_0: none_type_0}
    ordered_dict_0 = module_1.OrderedDict(**dict_0)
    var_0 = module_0.to_namedtuple(ordered_dict_0)
    var_1 = module_0.to_namedtuple(var_0)
    var_2 = module_0.to_namedtuple(dict_0)
    var_3 = module_0.to_namedtuple(var_0)
    str_1 = ""
    module_0.to_namedtuple(str_1)
