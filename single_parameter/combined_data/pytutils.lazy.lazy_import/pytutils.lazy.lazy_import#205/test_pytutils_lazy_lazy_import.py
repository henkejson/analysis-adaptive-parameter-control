# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pytutils.lazy.lazy_import as module_0


def test_case_0():
    str_0 = 'OW"Z DnJ%'
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        str_0, str_0, str_0
    )
    illegal_use_of_scope_replacer_0.__unicode__()


def test_case_1():
    none_type_0 = None
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        none_type_0, none_type_0, none_type_0
    )


def test_case_2():
    none_type_0 = None
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        none_type_0, none_type_0
    )
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_3():
    none_type_0 = None
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        none_type_0, none_type_0
    )
    var_0 = illegal_use_of_scope_replacer_0.__eq__(none_type_0)
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_4():
    none_type_0 = None
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        none_type_0, none_type_0
    )
    var_0 = illegal_use_of_scope_replacer_0.__eq__(illegal_use_of_scope_replacer_0)
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_5():
    bytes_0 = b"3}\xfa/\\{\x95E\x15Rx"
    module_0.ImportReplacer(bytes_0, bytes_0, bytes_0)


def test_case_6():
    import_processor_0 = module_0.ImportProcessor()


def test_case_7():
    str_0 = "~"
    module_0.lazy_import(str_0, str_0, str_0)


def test_case_8():
    str_0 = "4"
    module_0.lazy_import(str_0, str_0)


def test_case_9():
    var_0 = module_0.disallow_proxying()


def test_case_10():
    str_0 = "\n    Turn a function to a bound method on an instance\n\n    >>> class Foo(object):\n    ...     def __init__(self, x, y):\n    ...         self.x = x\n    ...         self.y = y\n    >>> foo = Foo(2, 3)\n    >>> my_unbound_method = lambda self: self.x * self.y\n    >>> bind(foo, my_unbound_method, 'multiply')\n    >>> foo.multiply()  # noinspection PyUnresolvedReferences\n    6\n\n    :param object instance: some object\n    :param callable func: unbound method (i.e. a function that takes `self` argument, that you now\n        want to be bound to this class as a method)\n    :param str as_name: name of the method to create on the object\n    "
    module_0.lazy_import(str_0, str_0)


def test_case_11():
    str_0 = "g"
    module_0.ImportReplacer(str_0, str_0, str_0, str_0)


def test_case_12():
    str_0 = "7(ef"
    module_0.lazy_import(str_0, str_0)


def test_case_13():
    str_0 = " "
    module_0.lazy_import(str_0, str_0)


def test_case_14():
    str_0 = "\n\n    >>> log = logging.getLogger(__name__)\n    >>> configure()\n    >>> log.info('test')\n\n    "
    module_0.lazy_import(str_0, str_0)


def test_case_15():
    bool_0 = True
    list_0 = [bool_0, bool_0, bool_0]
    module_0.ImportReplacer(bool_0, bool_0, list_0, bool_0, bool_0)


def test_case_16():
    str_0 = "\n    Turn a function to a bound method on an instance\n\n    >>> class Foo(object):\n    ...     def __init__(self, x, y):\n    ...         self.x = x\n    ...         self.y = y\n    >>> foo = Foo(2, 3)\n    >>> my_unbound_method = lambda self: self.x * self.y\n    >>> bind(foo, my_unbound_method, 'multiply')\n    >>> foo.multiply()  # noinspection PyUnresolvedReferences\n    6\n\n    :param object instance: some object\n    :param callable func: unbound method (i.e. a function that takes `self` argument, that you now\n        want to be bound to this class as a method)\n    :param str as_name: name of the method to create on the object\n    "
    dict_0 = {
        str_0: str_0,
        str_0: str_0,
        str_0: str_0,
        str_0: str_0,
        str_0: str_0,
        str_0: str_0,
    }
    import_replacer_0 = module_0.ImportReplacer(dict_0, str_0, dict_0, children=str_0)
    module_0.lazy_import(import_replacer_0, import_replacer_0)


def test_case_17():
    str_0 = "\n    Turn a function to a bound method on an instance\n\n    >>> class Foo(object):\n    ...     def __inMt__(self, x, y):\n    ...         self.x = x\n    ...         self.y = y\n    >>> foo = Foo(2, 3)\n    >>> my_unbound_method = lambda self: self.x * self.y\n    >>> bind(foo, my_unbound_method, 'multiply')\n    >>> foo.multiply((  # noinspection PyUnresolvedReferences\n    6\n\n    :param object instance: some object\n    :param callable func: unbound method (i.e. a function that takes `self` argument, that you now\n        want to be bound to this class as a method)\n    :param str as_name: name of the method to create on the object\n    "
    module_0.lazy_import(str_0, str_0)


def test_case_18():
    str_0 = "7    SJmulates nonlocal keywvrd in Python 2\n   D"
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0}
    import_replacer_0 = module_0.ImportReplacer(dict_0, str_0, dict_0, str_0)
    module_0.lazy_import(str_0, import_replacer_0)
