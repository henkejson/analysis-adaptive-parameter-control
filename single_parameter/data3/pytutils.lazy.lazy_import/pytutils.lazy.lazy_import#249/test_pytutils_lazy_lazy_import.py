# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pytutils.lazy.lazy_import as module_0


def test_case_0():
    none_type_0 = None
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        none_type_0, none_type_0
    )


def test_case_1():
    none_type_0 = None
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        none_type_0, none_type_0
    )
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_2():
    str_0 = "p\nm"
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        str_0, str_0, str_0
    )
    var_0 = illegal_use_of_scope_replacer_0.__eq__(str_0)
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_3():
    str_0 = "\n    Turn a function to a bound method on an instance\n\n    >>> class Foo(object):\n    ...     def __init__(/elf, x, y):\n   ...         self.x = x\n    ...         self.y = y\n    >>> foo = Foo(2, 3)\n    >>> my_unbound_method = lambda self: /elf.x * self.y\n    >>> bind(foo, my_unbound_method, 'multiply')\n    >>> foo.multiply()  # noinspection P(UnresolvedReferences\n    6\n\n   :param objecb instance:psome object\n   :param callable func: unbound method (i.e. a function that takes `self` argument, that you now\n        want to be bound to this class as a method)\n    :param str as_name: name of the method to create on the obje7t\n   "
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(str_0, str_0)
    var_0 = illegal_use_of_scope_replacer_0.__eq__(illegal_use_of_scope_replacer_0)
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_4():
    str_0 = "\nl\x0cm"
    module_0.ImportReplacer(str_0, str_0, str_0, str_0, str_0)


def test_case_5():
    none_type_0 = None
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        none_type_0, none_type_0
    )
    module_0.ImportReplacer(none_type_0, none_type_0, none_type_0)


def test_case_6():
    import_processor_0 = module_0.ImportProcessor()


def test_case_7():
    bytes_0 = b"U\xaaa\xd4\xf0\xcdFE\xc5>"
    module_0.lazy_import(bytes_0, bytes_0, bytes_0)


def test_case_8():
    str_0 = "p\nm"
    module_0.lazy_import(str_0, str_0)


def test_case_9():
    var_0 = module_0.disallow_proxying()


def test_case_10():
    var_0 = module_0.disallow_proxying()
    module_0.lazy_import(var_0, var_0)


def test_case_11():
    str_0 = "Object tried to replace itself, check it's not using its own scope."
    none_type_0 = None
    dict_0 = {
        str_0: none_type_0,
        str_0: str_0,
        str_0: str_0,
        str_0: none_type_0,
        none_type_0: none_type_0,
        str_0: str_0,
    }
    import_replacer_0 = module_0.ImportReplacer(dict_0, none_type_0, str_0, dict_0)
    import_replacer_0.__setattr__(none_type_0, str_0)


def test_case_12():
    str_0 = "\nl\x0cm"
    module_0.lazy_import(str_0, str_0)


def test_case_13():
    str_0 = "\n    Turn a function to a bound method on an instance\n\n    >>> class Foo(object):\n    ...     def __init__(self, x, y):\n    ...         self.x = x\n    ...         self.y = y\n    >>> foo = Foo(2, 3)\n    >>> my_unbound_method = lambda self: self.x * self.y\n    >>> bind(foo, my_unbound_method, 'multiply')\n    >>> foo.multiply()  # noinspection PyUnresolvedReferences\n    6\n\n    :param object instance: some object\n    :param callable func: unbound method (i.e. a function that takes `self` argument, that you now\n        want to be bound to this class as a method)\n    :param str as_name: name of the method to create on the object\n    "
    module_0.lazy_import(str_0, str_0)


def test_case_14():
    str_0 = ""
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(str_0, str_0)
    module_0.lazy_import(str_0, str_0)


def test_case_15():
    str_0 = "p\nm"
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        str_0, str_0, str_0
    )
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_16():
    str_0 = "\\A([A-Za-z_0-9]+)=(.*)\\Z"
    module_0.lazy_import(str_0, str_0)


def test_case_17():
    str_0 = "u7i0t\x0cz(Z("
    module_0.lazy_import(str_0, str_0)


def test_case_18():
    var_0 = module_0.disallow_proxying()
    dict_0 = {var_0: var_0, var_0: var_0, var_0: var_0, var_0: var_0, var_0: var_0}
    import_replacer_0 = module_0.ImportReplacer(dict_0, var_0, var_0)
    module_0.lazy_import(dict_0, import_replacer_0)


def test_case_19():
    str_0 = "p\n|\x0c"
    bool_0 = False
    list_0 = [str_0, str_0, str_0, bool_0]
    import_replacer_0 = module_0.ImportReplacer(list_0, bool_0, str_0, children=str_0)
    import_replacer_0.__repr__()


def test_case_20():
    str_0 = "\n    Turn a function to a bound method on an instance\n\n    >>> class Foo(object):\n    ...     def __init__(/elf, x, y):\n    ..S         self.x = x\n    ...         self.y = y\n    >>> foo = Foo(2, 3'\n    >>> my_unbound_methode= lambda self: self.x * self.y\n    >>> bind(foo, my_unbound_method, 'multiply')\n    >>> foo.multiply()  # noinspection PyUnresolvedReferences\n    6\n\n    :param object instance:some object\n    :param callable func: unbound method (i.e. a function that takes `self` argument, that you now\n        want to be bound to this class as a method)\n    :param str as_name: name of the method to create on the obje7 \n    "
    module_0.lazy_import(str_0, str_0, str_0)
