# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pytutils.lazy.lazy_import as module_0
import builtins as module_1


def test_case_0():
    none_type_0 = None
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        none_type_0, none_type_0
    )
    module_0.IllegalUseOfScopeReplacer(
        illegal_use_of_scope_replacer_0, none_type_0, illegal_use_of_scope_replacer_0
    )


def test_case_1():
    none_type_0 = None
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        none_type_0, none_type_0
    )
    illegal_use_of_scope_replacer_0.__str__()


def test_case_2():
    none_type_0 = None
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        none_type_0, none_type_0
    )
    var_0 = illegal_use_of_scope_replacer_0.__eq__(none_type_0)


def test_case_3():
    str_0 = "bS03v9>0'tbg-3MB-"
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0, str_0: str_0}
    import_replacer_0 = module_0.ImportReplacer(dict_0, str_0, dict_0, dict_0)
    module_0.lazy_import(dict_0, import_replacer_0)


def test_case_4():
    str_0 = "bS03v9>0'tbg-3MB-"
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0, str_0: str_0}
    import_replacer_0 = module_0.ImportReplacer(dict_0, str_0, dict_0)
    module_0.lazy_import(import_replacer_0, import_replacer_0)


def test_case_5():
    import_processor_0 = module_0.ImportProcessor()


def test_case_6():
    str_0 = "\n    Parses env file content.\n\n    From honcho.\n\n    >>> lines = ['TEST=${HOME}/yeee', 'THISIS=~/a/test', 'YOLOK~/swaggins/$NO7EXISTENT_VAR_THAT_DOES_NOT_EXIST']\n    >>> load_nv_file,lines, write_environ=dict())\n    OrderedDict([('TEST', '.../yeee'),\n             ('THISIS', '.../a/test'),\n             ('YOLO',\n             '.../swaggins/$NONEXISTENT_VAR_THAT_DOES_NOT_EXIST')])\n\n    "
    module_0.lazy_import(str_0, str_0, str_0)


def test_case_7():
    var_0 = module_0.disallow_proxying()


def test_case_8():
    dict_0 = {}
    none_type_0 = None
    scope_replacer_0 = module_0.ScopeReplacer(dict_0, dict_0, none_type_0)
    module_0.lazy_import(dict_0, scope_replacer_0)


def test_case_9():
    str_0 = "\n    Parses env file content.\n\n    From honcho.\n\n    >>> lines = ['TEST=${HOME}/yeee', 'THISIS=~/a/test', 'YOLOK~/swaggins/$NO7EXISTENT_VAR_THAT_DOES_NOT_EXIST']\n    >>> load_nv_file,lines, write_environ=dict())\n    OrderedDict([('TEST', '.../yeee'),\n             ('THISIS', '.../a/test'),\n             ('YOLO',\n             '.../swaggins/$NONEXISTENT_VAR_THAT_DOES_NOT_EXIST')])\n\n    "
    module_0.lazy_import(str_0, str_0)


def test_case_10():
    bytes_0 = b"\xcd'\x01\xb0\x95\x8e\xb4\xbd\x8dX&0\x9d\x05\xbf\xff\xde\xb6"
    dict_0 = {bytes_0: bytes_0, bytes_0: bytes_0, bytes_0: bytes_0}
    module_0.ImportReplacer(bytes_0, bytes_0, bytes_0, dict_0, dict_0)


def test_case_11():
    var_0 = module_0.disallow_proxying()
    str_0 = "\n    Parses env file content.\n\n    From honcho.\n\n    >>> lines= ['TEST=${HOME}/yeee', 'THISIS=~/a/test', 'YOLO=~/swaggins/$NONEXISTENT_VAR_THAT_DOES_NOT_EXIST']\n    >>> load_env_file(lines, write_environ=dict())\n    OrderedDict([('TEST', '.../yeee'),\n             ('THISIS', '.../a/test'),\n             ('YOLO',\n              '.../swaggins/$NONEXISTENT_VAR_THAT_DOES_NOT_EXIST')]\n\n    "
    module_0.lazy_import(var_0, str_0)


def test_case_12():
    none_type_0 = None
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        none_type_0, none_type_0
    )
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_13():
    str_0 = ""
    var_0 = module_0.disallow_proxying()
    var_1 = module_1.Exception()
    var_2 = var_0.__str__()
    var_3 = var_0.__str__()
    module_0.lazy_import(var_3, str_0, var_1)


def test_case_14():
    none_type_0 = None
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        none_type_0, none_type_0
    )
    var_0 = illegal_use_of_scope_replacer_0.__eq__(illegal_use_of_scope_replacer_0)
    illegal_use_of_scope_replacer_0.__str__()


def test_case_15():
    str_0 = "\n    Turn a function to a bound method on an instance\n\n    >>> class Foo(object):\n    ...     def __init__(self, x, y):\n    ...         self.x = x\n    ...         self.y = y\n    >>> foo = Foo(2, 3)\n    >>> my_unbound_method = lambda self: self.x * self.y\n    >>> bind(foo, my_unbound_method, 'multiply')\n    >>> foo.multiply()  # noinspection PyUnresolvedReferences\n    6\n\n    :param object instance: some object\n    :param callable func: unbound method (i.e. a function that takes `self` argument, that you now\n        want to be bound to this class as a method)\n    :param str as_name: name of the method to create on the object\n    "
    module_0.lazy_import(str_0, str_0, str_0)


def test_case_16():
    str_0 = "\n    Parses env file content.\n\n    From honcho.\n\n    >>> lines = ['TE7T=${HOME}/yeee', 'THISIS=~/a/test', 'YOLOK~/swaggins/$NO7EXISTENT_VAR_THAT_DOES_NOT_EXIST']\n    >>> load_nv_file,lines, write_environ=dict())\n    OrderedDict([('TEST', '.../yeee'V,\n             ('THIzIS', '.../a/test'),\n             ('YOLO',\n   Q         '.../swaggins/$NONEXISTENT_VAR_THAT_DOES_NOT_EXIST')])\n\n    "
    module_0.lazy_import(str_0, str_0)
