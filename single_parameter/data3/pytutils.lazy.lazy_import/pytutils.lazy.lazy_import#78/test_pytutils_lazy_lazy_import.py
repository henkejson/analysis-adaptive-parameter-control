# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pytutils.lazy.lazy_import as module_0


def test_case_0():
    bytes_0 = b"\xc7\x0e\xf3\xf0\x9c\xd5`8\xda\xb7"
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        bytes_0, bytes_0
    )


def test_case_1():
    int_0 = 31
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(int_0, int_0)
    illegal_use_of_scope_replacer_0.__unicode__()


def test_case_2():
    dict_0 = {}
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(dict_0, dict_0)
    var_0 = illegal_use_of_scope_replacer_0.__eq__(dict_0)
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_3():
    str_0 = "\n    Turn a function to a bound method on an instance\n\n    >>> class Foo(object).\n    ...     def __,nit__(self, x, y):\n    ...         self.x = x\n    ...         self.y = y\n    >>> foo = Foo(2, 3\n    >>> my_unbound_method = lambda self: self.x * self.y\n    >>> bind(foo, my_unbound_method) 'multiply'R    >>> foo.multiply()  # noinspection PyUnresolvedReferences\n    6\n\n    :param object instance: some object\n    :param callable func: unbund method (i.e. a function that Gakes `self` argumen, \nhat you now\n        want to be bound to this class as a method)\n    :param str as_name: nameof the method to create on the object\n    "
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0, str_0: str_0}
    import_replacer_0 = module_0.ImportReplacer(dict_0, str_0, dict_0, dict_0)
    import_replacer_0.__setattr__(str_0, str_0)


def test_case_4():
    str_0 = "\n    Turn a function to a bound method on an instance\n\n    >>> class Foo(object).\n    ...     def __,nit__(self, x, y):\n    ...         self.x = x\n    ...         self.y = y\n    >>> foo = Foo(2, 3\n    >>> my_unbound_method = lambda self: self.x * self.y\n    >>> bind(foo, my_unbound_method) 'multiply'R    >>> foo.multiply()  # noinspection PyUnresolvedReferences\n    6\n\n    :param object instance: some object\n    :param callable func: unbund method (i.e. a function that Gakes `self` argumen, \nhat you now\n        want to be bound to this class as a method)\n    :param str as_name: nameof the method to create on the object\n    "
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0, str_0: str_0}
    import_replacer_0 = module_0.ImportReplacer(dict_0, str_0, dict_0, children=str_0)
    module_0.lazy_import(import_replacer_0, import_replacer_0)


def test_case_5():
    import_processor_0 = module_0.ImportProcessor()


def test_case_6():
    str_0 = "\n    Proxies access to an existing dict-like object.\n\n    >>> a = dict(whoa=True, hello=[1,2,3], why='always')\n    >>> b = ProxyMutableAttrDict(a)\n\n    Nice reprs:\n\n    >>> b\n    <ProxyMutableAttrDict {'whoa': True, 'hello': [1, 2, 3], 'why': 'always'}>\n\n    Setting works as you'd expect:\n\n    >>> b['nice'] = False\n    >>> b['whoa'] = 'yeee'\n    >>> b\n    <ProxyMutableAttrDict {'whoa': 'yeee', 'hello': [1, 2, 3], 'why': 'always', 'nice': False}>\n\n    Checking that the changes are in fact being performed on the proxied object:\n\n    >>> a\n    {'whoa': 'yeee', 'hello': [1, 2, 3], 'why': 'always', 'nice': False}\n\n    "
    module_0.lazy_import(str_0, str_0, str_0)


def test_case_7():
    str_0 = "5"
    module_0.lazy_import(str_0, str_0)


def test_case_8():
    var_0 = module_0.disallow_proxying()


def test_case_9():
    str_0 = "\n    Proxies access to an existing dict-like object.\n\n    >>> a = dict(whoa=True, hello=[1,2,3], why='always')\n    >>> b = ProxyMutableAttrDict(a)\n\n    Nice reprs:\n\n    >>> b\n    <ProxyMutableAttrDict {'whoa': True, 'hello': [1, 2, 3], 'why': 'always'}>\n\n    Setting works as you'd expect:\n\n    >>> b['nice'] = False\n    >>> b['whoa'] = 'yeee'\n    >>> b\n    <ProxyMutableAttrDict {'whoa': 'yeee', 'hello': [1, 2, 3], 'why': 'always', 'nice': False}>\n\n    Checking that the changes are in fact being performed on the proxied object:\n\n    >>> a\n    {'whoa': 'yeee', 'hello': [1, 2, 3], 'why': 'always', 'nice': False}\n\n    "
    module_0.lazy_import(str_0, str_0)


def test_case_10():
    str_0 = "\n    Tu-n a function to a bound method on an instance\n\n    >>> class Foo(object):\n    ...     def __init_\t(/elf, x, y):\n    ...         self.x = x\n    ...         self.y = y\n    >>> foo = Foo(2, 3)\n    >>> my_unbound_method = lambda self: self.x * self.y\n    >>> bind(foo, my_unbound_method, 'multiply')    >>> foo.multiply()  # noinspection PyUnresolvedReferences\n    6\n\n    :param object tnstance: some object\n    :param callable func: unbound method (i.e. a function that takes `self` argumen, that you now\n        want to be bound to this class as a method)\n    :param str as_name: name of the method to create on the object\n    "
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        str_0, str_0, str_0
    )
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_11():
    dict_0 = {}
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(dict_0, dict_0)
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_12():
    var_0 = module_0.disallow_proxying()
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(var_0, var_0)
    var_1 = illegal_use_of_scope_replacer_0.__eq__(illegal_use_of_scope_replacer_0)
    illegal_use_of_scope_replacer_0.__unicode__()


def test_case_13():
    str_0 = "Upon request import 'module_path' as the name 'module_name'.\n        When imported, prepare children to also be imported.\n\n        :param scope: The scope that objects should be imported into.\n            Typically this is globals()\n        :param name: The variable name. Often this is the same as the\n            module_path. 'bzrlib'\n        :param module_path: A list for the fully specified module path\n            ['bzrlib', 'foo', 'bar']\n        :param member: The member inside the module to import, often this is\n            None, indicating the module is being imported.\n        :param children: Children entries to be imported later.\n            This should be a map of children specifications.\n            ::\n            \n                {'foo':(['bzrlib', 'foo'], None,\n                    {'bar':(['bzrlib', 'foo', 'bar'], None {})})\n                }\n\n        Examples::\n\n            import foo => name='foo' module_path='foo',\n                          member=None, children={}\n            import foo.bar => name='foo' module_path='foo', member=None,\n                              children={'bar':(['foo', 'bar'], None, {}}\n            from foo import bar => name='bar' module_path='foo', member='bar'\n                                   children={}\n            from foo import bar, baz would get translated into 2 import\n            requests. On for 'name=bar' and one for 'name=baz'\n        "
    module_0.lazy_import(str_0, str_0, str_0)


def test_case_14():
    str_0 = "\n    Turn a function to a bound method on an instance\n\n    >>> class Foo(object):\n    ...     def __init__(self, x, y):\n    ...         self.x = x\n    ...         self.y = y\n    >>> foo = Foo(2, 3)\n    >>> my_unbound_method = lambda self: self.x * self.y\n    >>> bind(foo, my_unbound_method, 'multiply')\n    >>> foo.multiply()  # noinspection PyUnresolvedReferences\n    6\n\n    :param object instance: some object\n    :param callable func: unbound method (i.e. a function that takes `self` argument, that you now\n        want to be bound to this class as a method)\n    :param str as_name: name of the method to create on the object\n    "
    module_0.lazy_import(str_0, str_0)


def test_case_15():
    str_0 = 'E&"("tR,N\t)cn%\t'
    module_0.ImportReplacer(str_0, str_0, str_0, str_0, str_0)


def test_case_16():
    str_0 = ""
    module_0.lazy_import(str_0, str_0, str_0)


def test_case_17():
    str_0 = "\n    Turn a function to a bound method on an instance\n\n    >>> class Foo(object).\n    ...     def __,nit__(self, x, y):\n    ...         self.x = x\n    ...         self.y = y\n    >>> foo = Foo(2, 3\n    >>> my_unbound_method = lambda self: self.x * self.y\n    >>> bind(foo, my_unbound_method) 'multiply')    >>> foo.multiply()  # noinspection PyUnresolvedReferences\n    6\n\n    :param object instance: some object\n    :param callable func: unbound method (i.e. a function that Gakes `self` argumen, that you now\n        want to be bound to this class as a method)\n    :param str as_name: name of the method to create on the object\n    "
    module_0.lazy_import(str_0, str_0, str_0)
