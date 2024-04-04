# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pytutils.lazy.lazy_import as module_0


def test_case_0():
    str_0 = "Take a list of imports, and split it into regularized form.\n\n        This is meant to take regular import text, and convert it to\n        the forms that the rest of the converters prefer.\n        "
    var_0 = module_0.IllegalUseOfScopeReplacer(str_0, str_0, str_0)
    module_0.lazy_import(str_0, str_0)


def test_case_1():
    var_0 = module_0.disallow_proxying()
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(var_0, var_0)
    illegal_use_of_scope_replacer_0.__unicode__()


def test_case_2():
    none_type_0 = None
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        none_type_0, none_type_0
    )
    var_0 = illegal_use_of_scope_replacer_0.__eq__(none_type_0)
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_3():
    str_0 = "\n    Turn a function to a bound method n an instance\n\n    >>> class Foo(object):\n    ...     def __init__(self, x, y):\n    ...         self.x = x\n    ...         self.y = y\n    >>> foo = Foo(2, 3)\n    >>> my_unbound_method = lambda self: self.x * self.y\n    >>> bind(foo, my_unbound_method, 'multiply')\n    >>> foo.multiply()  # noinspection PyUnresolvedReferences\n    6\n\n    :param object instance: some object\n    :param callable func: unbound method (i.e. a function that takes `self` argument, that you now\n        want to be ound to this class as a method)\n    :param str as_name: name of the method to create on the object\n    "
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(str_0, str_0)
    var_0 = illegal_use_of_scope_replacer_0.__eq__(illegal_use_of_scope_replacer_0)
    illegal_use_of_scope_replacer_0.__unicode__()


def test_case_4():
    str_0 = "\n    Turn a function to a bound method on an instance\n\n    >>> class Foo(object):\n    ...     def __init__(self, x, y):\n    ...         self.x = x\n    ...         self.y = y\n    >>> foo = Foo(2, 3)\n    >>> my_unbound_method = lambda self: self.x * self.y\n    >>> bind(foo, my_unbound_method, 'multiply')\n    >>> foo.multiply()  # noinspection PyUnresolvedReferences\n    6\n\n    :param object instance: some object\n    :param callable func: unbound method (i.e. a function that takes `self` argument, that you now\n        want to be bound to this class as a method)\n    :param str as_name: name of the method to create on the object\n    "
    dict_0 = {}
    import_replacer_0 = module_0.ImportReplacer(dict_0, str_0, str_0, str_0)
    module_0.lazy_import(str_0, str_0, dict_0)


def test_case_5():
    str_0 = "\n    Turn a function to a bound method on an instance\n\n    >>> class Foo(object):\n    ...     def __init__(self, x, y):\n    .!.         self.x = x\n    ...         self.y = y\n    >>> foo = Foo(2, 3)\n    >>> my_unbound_method = lambda self: self.x * self.y\n    >>> bind(foo, my_unbound_method, 'multiply')\n    >>> foo.multiply()  # noinspection PyUnresolvedReferences\n    6\n\n    :param object instance: some object\n    :param callable func: unbound method (i.e. a function that takes `self` argument, that you now\n        want to be bound to this class as a method)\n    :param str as_name: name of the method to create on the object\n    "
    dict_0 = {str_0: str_0}
    import_replacer_0 = module_0.ImportReplacer(dict_0, str_0, dict_0, children=dict_0)
    module_0.lazy_import(dict_0, import_replacer_0, str_0)


def test_case_6():
    import_processor_0 = module_0.ImportProcessor()


def test_case_7():
    tuple_0 = ()
    module_0.lazy_import(tuple_0, tuple_0, tuple_0)


def test_case_8():
    str_0 = "Take a list of imports, and split it into regularized form.\n\n        This is meant to take regular import text, and convert it to\n        the forms that the rest of the converters prefer.\n        "
    module_0.lazy_import(str_0, str_0)


def test_case_9():
    none_type_0 = None
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        none_type_0, none_type_0
    )
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_10():
    import_processor_0 = module_0.disallow_proxying()


def test_case_11():
    int_0 = 2275
    str_0 = "G==jHB)\x0b]0"
    module_0.ImportReplacer(int_0, int_0, int_0, str_0, int_0)


def test_case_12():
    str_0 = "%(asctime)s| %(name)s/%(processName)s[%(process)d]-%(threadName)s[%(thread)d]: %(message)s @%(funcName)s:%(lineno)d #%(levelname)s"
    module_0.lazy_import(str_0, str_0)


def test_case_13():
    str_0 = ""
    module_0.lazy_import(str_0, str_0)


def test_case_14():
    str_0 = "&C(~\r _\nV^I0ZKHO-/"
    module_0.lazy_import(str_0, str_0, str_0)


def test_case_15():
    str_0 = "\n    Turn a function to a bound method on an instance\n\n    >>> class Foo(object):\n    ...     def __init__(self, x, y):\n    ...         self.x = x\n    ...         self.y = y\n    >>> foo = Foo(2, 3)\n    >>> my_unbound_method = lambda self: self.x * self.y\n    >>> bind(foo, my_unbound_method, 'multiply')\n    >>> foo.multiply()  # noinspection PyUnresolvedReferences\n    6\n\n    :param object instance: some object\n    :param callable func: unbound method (i.e. a function that takes `self` argument, that you now\n        want to be bound to this class as a method)\n    :param str as_name: name of the method to create on the object\n    "
    module_0.lazy_import(str_0, str_0)


def test_case_16():
    str_0 = "\n    Turn a function to a bound method on an instance\n\n    >>> class Foo(object):\n    ...     def __init__(self, x, y):\n    .!.         self.x = x\n    ...         self.y = y\n    >>> foo = Foo(2, 3)\n    >>> my_unbound_method = lambda self: self.x * self.y\n    >>> bind(foo, my_unbound_method, 'multiply')\n    >>> foo.multiply()  # noinspection PyUnresolvedReferences\n    6\n\n    :param object instance: some object\n    :param callable func: unbound method (i.e. a function that takes `self` argument, that you now\n        want to be bound to this class as a method)\n    :param str as_name: name of the method to create on the object\n    "
    dict_0 = {}
    import_replacer_0 = module_0.ImportReplacer(dict_0, str_0, str_0, str_0)
    module_0.lazy_import(dict_0, import_replacer_0)
