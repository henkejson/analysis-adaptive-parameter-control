# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import flutils.namedtupleutils as module_0
import collections as module_1
import builtins as module_2


def test_case_0():
    float_0 = -3118.1
    module_0.to_namedtuple(float_0)


def test_case_1():
    bytes_0 = b""
    bool_0 = True
    list_0 = [bytes_0, bool_0, bytes_0]
    var_0 = module_0.to_namedtuple(list_0)


def test_case_2():
    str_0 = "Lazy import a python module.\n\n    Args:\n        name (:obj:`str`): specifies what module to import in absolute or\n            relative terms (e.g. either ``pkg.mod`` or ``..mod``).\n        package (:obj:`str`, optional): If ``name`` is specified in relative\n            terms, then the ``package`` argument must be set to the name of the\n            package which is to act as the anchor for resolving the package\n            name.  Defaults to ``None``.\n\n    Raises:\n        ImportError: if the given ``name`` and ``package`` can not be loaded.\n\n    :rtype:\n        :obj:`Module <types.ModuleType>`\n\n        * The lazy imported module with the execution of it's loader postponed\n          until an attribute accessed.\n\n    .. Warning:: For projects where startup time is critical, this function\n        allows for potentially minimizing the cost of loading a module if it\n        is never used. For projects where startup time is not essential then\n        use of this function is heavily discouraged due to error messages\n        created during loading being postponed and thus occurring out of\n        context.\n\n    Examples:\n\n        >>> from flutils.moduleutils import lazy_import_module\n        >>> module = lazy_import_module('mymodule')\n\n        Relative import:\n\n        >>> module = lazy_import_module('.mysubmodule', package='mymodule')\n    "
    dict_0 = {str_0: str_0}
    var_0 = module_0.to_namedtuple(dict_0)


def test_case_3():
    ordered_dict_0 = module_1.OrderedDict()
    var_0 = module_0.to_namedtuple(ordered_dict_0)


def test_case_4():
    bytes_0 = b""
    module_0.to_namedtuple(bytes_0)


def test_case_5():
    list_0 = []
    var_0 = module_0.to_namedtuple(list_0)


def test_case_6():
    str_0 = "Lazy import a python module.\n\n    Args:\n        name (:obj:`str`): specifies what module to import in absolute or\n            relative terms (e.g. either ``pkg.mod`` or ``..mod``).\n        package (:obj:`str`, optional): If ``name`` is specified in relative\n            terms, then the ``package`` argument must be set to the name of the\n            package which is to act as the anchor for resolving the package\n            name.  Defaults to ``None``.\n\n    Raises:\n        ImportError: if the given ``name`` and ``package`` can not be loaded.\n\n    :rtype:\n        :obj:`Module <types.ModuleType>`\n\n        * The lazy imported module with the execution of it's loader postponed\n          until an attribute accessed.\n\n    .. Warning:: For projects where startup time is critical, this function\n        allows for potentially minimizing the cost of loading a module if it\n        is never used. For projects where startup time is not essential then\n        use of this function is heavily discouraged due to error messages\n        created during loading being postponed and thus occurring out of\n        context.\n\n    Examples:\n\n        >>> from flutils.moduleutils import lazy_import_module\n        >>> module = lazy_import_module('mymodule')\n\n        Relative import:\n\n        >>> module = lazy_import_module('.mysubmodule', package='mymodule')\n    "
    dict_0 = {str_0: str_0}
    var_0 = module_0.to_namedtuple(dict_0)
    var_1 = module_0.to_namedtuple(var_0)
    var_2 = module_0.to_namedtuple(var_1)
    var_3 = module_0.to_namedtuple(var_2)
    dict_1 = {var_2: var_0, var_0: var_3}
    var_4 = module_0.to_namedtuple(dict_1)
    var_5 = module_0.to_namedtuple(dict_0)
    var_6 = module_0.to_namedtuple(var_5)
    var_7 = module_0.to_namedtuple(var_4)


def test_case_7():
    str_0 = "has_callables"
    dict_0 = {str_0: str_0}
    var_0 = module_0.to_namedtuple(dict_0)


def test_case_8():
    ordered_dict_0 = module_1.OrderedDict()
    var_0 = module_0.to_namedtuple(ordered_dict_0)
    var_1 = module_0.to_namedtuple(var_0)
    var_2 = module_0.to_namedtuple(var_1)


def test_case_9():
    tuple_0 = ()
    var_0 = module_0.to_namedtuple(tuple_0)


def test_case_10():
    str_0 = "has_callables"
    dict_0 = {str_0: str_0}
    var_0 = module_0.to_namedtuple(dict_0)
    var_1 = module_0.to_namedtuple(var_0)


def test_case_11():
    bytes_0 = b"\xbc\x8bK\xb7\xc2\x1a\xba\xd3\xd6\xfb"
    dict_0 = {bytes_0: bytes_0, bytes_0: bytes_0, bytes_0: bytes_0, bytes_0: bytes_0}
    list_0 = [dict_0]
    ordered_dict_0 = module_1.OrderedDict(*list_0)
    module_0.to_namedtuple(ordered_dict_0)


def test_case_12():
    str_0 = " ymEVL"
    dict_0 = {str_0: str_0}
    bool_0 = False
    tuple_0 = (bool_0,)
    tuple_1 = (str_0, dict_0, tuple_0)
    tuple_2 = (tuple_1, bool_0)
    var_0 = module_0.to_namedtuple(tuple_2)
    var_1 = module_0.to_namedtuple(var_0)
    ordered_dict_0 = module_1.OrderedDict()
    var_2 = module_0.to_namedtuple(ordered_dict_0)
    var_3 = module_0.to_namedtuple(tuple_2)
    var_4 = module_0.to_namedtuple(tuple_2)
    var_5 = module_0.to_namedtuple(var_2)
    object_0 = module_2.object()
    var_6 = module_0.to_namedtuple(tuple_1)
    var_7 = module_0.to_namedtuple(var_2)
    var_8 = module_0.to_namedtuple(tuple_0)
    var_9 = module_0.to_namedtuple(var_8)
    object_1 = module_2.object()
    module_0.to_namedtuple(object_1)
