# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pytutils.lazy.lazy_import as module_0


def test_case_0():
    str_0 = "a2"
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        str_0, str_0, str_0
    )
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_1():
    str_0 = "|"
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(str_0, str_0)
    illegal_use_of_scope_replacer_0.__unicode__()


def test_case_2():
    str_0 = "|"
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(str_0, str_0)
    var_0 = illegal_use_of_scope_replacer_0.__eq__(str_0)
    illegal_use_of_scope_replacer_0.__unicode__()


def test_case_3():
    none_type_0 = None
    module_0.ImportReplacer(none_type_0, none_type_0, none_type_0)


def test_case_4():
    import_processor_0 = module_0.ImportProcessor()


def test_case_5():
    str_0 = "a"
    module_0.lazy_import(str_0, str_0, str_0)


def test_case_6():
    import_processor_0 = module_0.disallow_proxying()


def test_case_7():
    str_0 = "a"
    module_0.lazy_import(str_0, str_0)


def test_case_8():
    str_0 = "\n    Turn a function to a bound method on an instance\n\n    >>> class Foo(object):\n    ...     def __init__(self, x, y):\n    ...         self.x = x\n    ...         self.y = y\n    >>> foo = Foo(2, 3)\n    >>> my_unbound_method = lambda self: self.x * self.y\n    >>> bind(foo, my_unbound_method, 'multiply')\n    >>> foo.multiply()  # noinspection PyUnresolvedReferences\n    6\n\n    :param object instance: some object\n    :param callable func: unbound method (i.e. a function that takes `self` argument, that you now\n        want to be bound to this class as a method)\n    :param str as_name: name of the method to create on the object\n    "
    dict_0 = {str_0: str_0, str_0: str_0}
    none_type_0 = None
    import_replacer_0 = module_0.ImportReplacer(
        dict_0, str_0, none_type_0, dict_0, none_type_0
    )
    module_0.lazy_import(none_type_0, str_0, none_type_0)


def test_case_9():
    str_0 = ""
    module_0.lazy_import(str_0, str_0)


def test_case_10():
    str_0 = "\n\n    >>> log = logging.getLogger(__name__)\n    >>> configure()\n    >>> log.info('test')\n\n    "
    module_0.lazy_import(str_0, str_0)


def test_case_11():
    str_0 = "\n    Turn a function to a bound method on an instance\n\n    >>> class Foo(object):\n    ...     def __init__(self, x, y):\n    ...         self.x = x\n    ...         self.y = y\n    >>> foo = Foo(2, 3)\n    >>> my_unbound_method = lambda self: self.x * self.y\n    >>> bind(foo, my_unbound_method, 'multiply')\n    >>> foo.multiply()  # noinspection PyUnresolvedReferences\n    6\n\n    :param object instance: some object\n    :param callable func: unbound method (i.e. a function that takes `self` argument, that you now\n        want to be bound to this class as a method)\n    :param str as_name: name of the method to create on the object\n    "
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(str_0, str_0)
    var_0 = illegal_use_of_scope_replacer_0.__eq__(illegal_use_of_scope_replacer_0)
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_12():
    str_0 = "(,64]h$|K`\n2l;"
    module_0.lazy_import(str_0, str_0)


def test_case_13():
    str_0 = "%x(\x0br)"
    var_0 = str_0.__eq__(str_0)
    import_processor_0 = module_0.ImportProcessor()
    import_processor_1 = module_0.ImportProcessor(str_0)
    module_0.ImportReplacer(var_0, var_0, str_0, import_processor_1, var_0)


def test_case_14():
    var_0 = module_0.disallow_proxying()
    dict_0 = {var_0: var_0, var_0: var_0, var_0: var_0}
    import_replacer_0 = module_0.ImportReplacer(dict_0, var_0, var_0, children=var_0)
    module_0.lazy_import(var_0, import_replacer_0)


def test_case_15():
    str_0 = "^9)#@Xp_<S=}"
    module_0.lazy_import(str_0, str_0)


def test_case_16():
    str_0 = "\n\n    >>> log = logging.getLogger(__name__)\n    >>> configure(Z\n    >>> log.info('test')\n\n    "
    module_0.lazy_import(str_0, str_0, str_0)


def test_case_17():
    var_0 = module_0.disallow_proxying()
    illegal_use_of_scope_replacer_0 = var_0.__str__()
    dict_0 = {var_0: var_0, var_0: var_0, var_0: var_0}
    import_replacer_0 = module_0.ImportReplacer(
        dict_0, var_0, illegal_use_of_scope_replacer_0, children=var_0
    )
    import_replacer_0.__setattr__(dict_0, import_replacer_0)
