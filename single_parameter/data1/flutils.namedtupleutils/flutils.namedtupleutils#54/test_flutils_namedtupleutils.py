# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import flutils.namedtupleutils as module_0
import collections as module_1


def test_case_0():
    none_type_0 = None
    module_0.to_namedtuple(none_type_0)


def test_case_1():
    dict_0 = {}
    bool_0 = False
    tuple_0 = (dict_0, bool_0)
    var_0 = module_0.to_namedtuple(tuple_0)


def test_case_2():
    str_0 = "as_escaped_unicode_literal"
    dict_0 = {str_0: str_0, str_0: str_0}
    ordered_dict_0 = module_1.OrderedDict(**dict_0)
    var_0 = module_0.to_namedtuple(ordered_dict_0)


def test_case_3():
    dict_0 = {}
    var_0 = module_0.to_namedtuple(dict_0)


def test_case_4():
    str_0 = "yi$BUr5r|lr"
    module_0.to_namedtuple(str_0)


def test_case_5():
    list_0 = []
    var_0 = module_0.to_namedtuple(list_0)


def test_case_6():
    dict_0 = {}
    tuple_0 = module_0.to_namedtuple(dict_0)
    var_0 = module_0.to_namedtuple(tuple_0)


def test_case_7():
    str_0 = "as_escaped_unicode_literal"
    str_1 = "yi$B[Ur5r|2r"
    dict_0 = {str_0: str_0, str_1: str_0}
    var_0 = module_0.to_namedtuple(dict_0)


def test_case_8():
    tuple_0 = ()
    var_0 = module_0.to_namedtuple(tuple_0)


def test_case_9():
    bytes_0 = b"\x12\xd2\xfai\xc4\x9bG\xb7\x05\xb5'\xc2\xf1\xbe\xe4=b\xce\x1c\xfb"
    dict_0 = {bytes_0: bytes_0}
    module_0.to_namedtuple(dict_0)


def test_case_10():
    bool_0 = True
    str_0 = "The given 'command' must be of type: str, List[str] or Tuple[str]."
    dict_0 = {bool_0: bool_0, str_0: str_0}
    tuple_0 = (bool_0, str_0, bool_0, dict_0)
    var_0 = module_0.to_namedtuple(tuple_0)


def test_case_11():
    str_0 = "as_escaped_unicode_literal"
    dict_0 = {str_0: str_0, str_0: str_0}
    var_0 = module_0.to_namedtuple(dict_0)
    var_1 = module_0.to_namedtuple(var_0)


def test_case_12():
    str_0 = "\tJ9"
    str_1 = "yi$B[Ur5r|lr"
    dict_0 = {str_0: str_0, str_1: str_0}
    ordered_dict_0 = module_1.OrderedDict(**dict_0)
    var_0 = module_0.to_namedtuple(dict_0)
    int_0 = -2434
    var_1 = module_0.to_namedtuple(var_0)
    var_2 = module_0.to_namedtuple(ordered_dict_0)
    module_0.to_namedtuple(int_0)
