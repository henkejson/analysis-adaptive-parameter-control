# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pytutils.lazy.lazy_import as module_0


def test_case_0():
    float_0 = 2670.9292
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        float_0, float_0, float_0
    )
    illegal_use_of_scope_replacer_0.__str__()


def test_case_1():
    str_0 = "1%:U\x0c6' W\nk56\n/S[|p"
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(str_0, str_0)


def test_case_2():
    none_type_0 = None
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        none_type_0, none_type_0, none_type_0
    )
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_3():
    str_0 = "\x0b\x0c]V4@/jng"
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(str_0, str_0)
    var_0 = illegal_use_of_scope_replacer_0.__eq__(str_0)
    module_0.lazy_import(str_0, str_0)


def test_case_4():
    str_0 = "]J.T\x0b0vPF8],W#/HyX(c"
    dict_0 = {str_0: str_0}
    import_replacer_0 = module_0.ImportReplacer(dict_0, str_0, dict_0, str_0)
    module_0.lazy_import(import_replacer_0, import_replacer_0)


def test_case_5():
    bytes_0 = b'"`\xa6\xddN\xfb\xbdTbF\x91'
    list_0 = [bytes_0, bytes_0]
    module_0.ImportReplacer(list_0, list_0, bytes_0)


def test_case_6():
    import_processor_0 = module_0.ImportProcessor()


def test_case_7():
    bool_0 = False
    import_processor_0 = module_0.ImportProcessor(bool_0)


def test_case_8():
    str_0 = ".mjM*D-3 $+\x0bB}:4"
    module_0.lazy_import(str_0, str_0)


def test_case_9():
    str_0 = ".mjM#?*D-3 $+\x0bB}:*4"
    str_1 = "Upon request import 'module_path' as the name 'module_name'.\n        When imported, prepare children to also be imported.\n\n        :param scope: The scope that objects should be imported into.\n            Typically this is globals()\n        :param name: The variable name. Often this is the same as the\n            module_path. 'bzrlib'\n        :param module_path: A list for the fully specified module path\n            ['bzrlib', 'foo', 'bar']\n        :param member: The member inside the module to import, often this is\n            None, indicating the module is being imported.\n        :param children: Children entries to be imported later.\n            This should be a map of children specifications.\n            ::\n            \n                {'foo':(['bzrlib', 'foo'], None,\n                    {'bar':(['bzrlib', 'foo', 'bar'], None {})})\n                }\n\n        Examples::\n\n            import foo => name='foo' module_path='foo',\n                          member=None, children={}\n            import foo.bar => name='foo' module_path='foo', member=None,\n                              children={'bar':(['foo', 'bar'], None, {}}\n            from foo import bar => name='bar' module_path='foo', member='bar'\n                                   children={}\n            from foo import bar, baz would get translated into 2 import\n            requests. On for 'name=bar' and one for 'name=baz'\n        "
    module_0.lazy_import(str_0, str_1)


def test_case_10():
    var_0 = module_0.disallow_proxying()


def test_case_11():
    str_0 = "\n    Turn a function to a bound method on an instance\n\n    >>> class Foo(object):\n    ...     def __init__(self, x, y):\n    ...         self.x = x\n    ...         self.y = y\n    >>> foo = Foo(2, 3)\n    >>> my_unbound_method = lambda self: self.x * self.y\n    >>> bind(foo, my_unbound_method, 'multiply')\n    >>> foo.multiply()  # noinspection PyUnresolvedReferences\n    6\n\n    :param object instance: some object\n    :param callable func: unbound method (i.e. a function that takes `self` argument, that you now\n        want to be bound to this class as a method)\n    :param str as_name: name of the method to create on the object\n    "
    module_0.lazy_import(str_0, str_0)


def test_case_12():
    str_0 = ".mjM#?*D-3 $+\x0bB}:4"
    module_0.lazy_import(str_0, str_0)


def test_case_13():
    str_0 = "Zs"
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(str_0, str_0)
    var_0 = illegal_use_of_scope_replacer_0.__eq__(illegal_use_of_scope_replacer_0)
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_14():
    str_0 = ""
    module_0.lazy_import(str_0, str_0, str_0)


def test_case_15():
    var_0 = module_0.disallow_proxying()
    var_1 = var_0.__eq__(var_0)
    module_0.ImportReplacer(var_1, var_1, var_0, var_1, var_1)


def test_case_16():
    str_0 = "tk7mfC<4o{n$,\tKu"
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0, str_0: str_0}
    import_replacer_0 = module_0.ImportReplacer(dict_0, str_0, dict_0)
    module_0.lazy_import(dict_0, import_replacer_0)


def test_case_17():
    str_0 = "Create lazy imports for all of the imports in text.\n\n    This is typically used as something like::\n\n        from bzrlib.lazy_import import lazy_import\n        lazy_import(globals(), '''\n        from bzrlib import (\n            foo,\n            bar,\n            baz,\n            )\n        import bzrlib.branch\n        import bzrlib.transport\n        ''')\n\n    Then 'foo, bar, baz' and 'bzrlib' will exist as lazy-loaded\n    objects which will be replaced with a real object on first use.\n\n    In general, it is best to only load modules in this way. This is\n    because other objects (functions/classes/variables) are frequently\n    used without accessing a member, which means we cannot tell they\n    have been used.\n    "
    module_0.lazy_import(str_0, str_0)
