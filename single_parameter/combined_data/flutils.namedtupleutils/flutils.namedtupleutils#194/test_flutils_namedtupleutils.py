# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import flutils.namedtupleutils as module_0
import collections as module_1
import builtins as module_2


def test_case_0():
    none_type_0 = None
    module_0.to_namedtuple(none_type_0)


def test_case_1():
    bool_0 = True
    tuple_0 = (bool_0,)
    var_0 = module_0.to_namedtuple(tuple_0)


def test_case_2():
    float_0 = 2448.29
    dict_0 = {float_0: float_0, float_0: float_0, float_0: float_0}
    list_0 = [dict_0, dict_0, float_0]
    var_0 = module_0.to_namedtuple(list_0)


def test_case_3():
    dict_0 = {}
    var_0 = module_0.to_namedtuple(dict_0)


def test_case_4():
    bytes_0 = b"\xc0\x80\xa5y"
    module_0.to_namedtuple(bytes_0)


def test_case_5():
    list_0 = []
    var_0 = module_0.to_namedtuple(list_0)


def test_case_6():
    str_0 = "KHlWo"
    none_type_0 = None
    dict_0 = {str_0: none_type_0}
    ordered_dict_0 = module_1.OrderedDict(**dict_0)
    list_0 = [ordered_dict_0, dict_0, ordered_dict_0]
    var_0 = module_0.to_namedtuple(list_0)


def test_case_7():
    ordered_dict_0 = module_1.OrderedDict()
    var_0 = module_0.to_namedtuple(ordered_dict_0)


def test_case_8():
    str_0 = "KHlWo"
    none_type_0 = None
    dict_0 = {str_0: none_type_0}
    ordered_dict_0 = module_1.OrderedDict(**dict_0)
    list_0 = [ordered_dict_0, dict_0, ordered_dict_0, str_0]
    var_0 = module_0.to_namedtuple(list_0)
    var_1 = module_0.to_namedtuple(var_0)


def test_case_9():
    ordered_dict_0 = module_1.OrderedDict()
    var_0 = module_0.to_namedtuple(ordered_dict_0)
    var_1 = module_0.to_namedtuple(var_0)


def test_case_10():
    bytes_0 = b"j\x08"
    dict_0 = {bytes_0: bytes_0}
    module_0.to_namedtuple(dict_0)


def test_case_11():
    tuple_0 = ()
    var_0 = module_0.to_namedtuple(tuple_0)


def test_case_12():
    bool_0 = False
    dict_0 = {bool_0: bool_0}
    str_0 = ",Xo7%\x0cR%#"
    dict_1 = {str_0: str_0, str_0: dict_0, str_0: bool_0}
    ordered_dict_0 = module_1.OrderedDict(**dict_1)
    var_0 = module_0.to_namedtuple(dict_0)
    bool_1 = False
    list_0 = [dict_0, ordered_dict_0, bool_1, dict_0]
    var_1 = module_0.to_namedtuple(list_0)
    bool_2 = False
    dict_2 = {bool_2: bool_2}
    var_2 = module_0.to_namedtuple(dict_2)
    var_3 = module_0.to_namedtuple(var_2)
    var_4 = module_0.to_namedtuple(var_3)
    var_5 = module_0.to_namedtuple(dict_2)
    var_6 = module_0.to_namedtuple(dict_0)
    var_7 = module_0.to_namedtuple(var_2)
    var_8 = module_0.to_namedtuple(var_2)
    module_0.to_namedtuple(bool_2)


def test_case_13():
    str_0 = "KHlWo"
    none_type_0 = None
    dict_0 = {str_0: none_type_0}
    ordered_dict_0 = module_1.OrderedDict(**dict_0)
    list_0 = module_0.to_namedtuple(dict_0)
    var_0 = module_0.to_namedtuple(list_0)


def test_case_14():
    bool_0 = True
    dict_0 = {bool_0: bool_0}
    str_0 = "\x0cm"
    dict_1 = {str_0: str_0, str_0: dict_0, str_0: bool_0}
    ordered_dict_0 = module_1.OrderedDict(**dict_1)
    bool_1 = False
    list_0 = [dict_0, ordered_dict_0, bool_1, dict_0]
    var_0 = module_0.to_namedtuple(list_0)
    bool_2 = False
    dict_2 = {bool_2: bool_2}
    var_1 = module_0.to_namedtuple(dict_2)
    ordered_dict_1 = module_1.OrderedDict(*var_1, **ordered_dict_0)
    var_2 = module_0.to_namedtuple(ordered_dict_1)
    object_0 = module_2.object(*var_1)
    ordered_dict_2 = module_1.OrderedDict()
    tuple_0 = (bool_2, bool_2, bool_2, dict_2)
    list_1 = [tuple_0, ordered_dict_1, str_0, dict_0]
    list_2 = [list_1, ordered_dict_1]
    var_3 = module_0.to_namedtuple(list_2)
    var_4 = module_0.to_namedtuple(var_0)
    module_0.to_namedtuple(bool_0)
