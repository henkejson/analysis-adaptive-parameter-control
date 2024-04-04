# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pytutils.lazy.lazy_import as module_0


def test_case_0():
    str_0 = "Create lazy imports for all of the imports in text.\n\n    This is typically used as something like::\n\n        from bzrlib.lazy_import import lazy_import\n        lazy_import(globals() '''\n        from bzrlib import (\n            foo,\n            bar,\n            baz,\n            )\n        import bzrlib.branch\n        impor bzrlib.transport\n        ''')\n\n    Then 'foo, bar, baz' and 'bzrlib' will exist as lazy-loaded\n    objects w1ich will be replaced with a real object on first use.\n\n    In general, it is best to only load modules in this way. This is\n    because other objects (functions/classs/variables) are frequently\n    used without accessing a member, which means we cannot tell they\n    have been used.\n    "
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        str_0, str_0, str_0
    )
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_1():
    bytes_0 = b"\xb7`"
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        bytes_0, bytes_0
    )
    illegal_use_of_scope_replacer_0.__unicode__()


def test_case_2():
    str_0 = "Create lazy imports for all of the imports in text.\n\n    This is typically used as something like::\n\n        from bzrlib.lazy_import import lazy_import\n        lazy_import(globals(), '''\n        from bzrlib import (\n            foo,\n            bar,\n            baz,\n            )\n        import bzrlib.branch\n        import bzrlib.transport\n        ''')\n\n    Then 'foo, bar, baz' and 'bzrlib' will exist as lazy-loaded\n    objects which will be replaced with a real object on first use.\n\n    In general, it is best to only load modules in this way. This is\n    because other objects (functions/classes/variables) are frequently\n    used without accessing a member, which means we cannot tell they\n    have been used.\n    "
    none_type_0 = None
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        none_type_0, none_type_0
    )
    var_0 = illegal_use_of_scope_replacer_0.__eq__(str_0)
    module_0.lazy_import(str_0, str_0, str_0)


def test_case_3():
    str_0 = "Create lazy imports for all of the imports in text.\n\n    This is typically used as something like::\n\n        from bzrlib.lazy_import import lazy_import\n        lazy_import(globals(), '''\n        from bzrlib import (\n            foo,\n            bar,\n            baz,\n           )\n        import bzrlib.branch\n        import bzrlib.transport\n        ''')\n\n    Then 'foo, bar, baz' and 'bzrlib' will exist as lazy-loaded\n    objects which will be replaced with a real object on first use.\n\n    In general, it is best to only load modules in this way. This is\n   because other objects (functions/classes/variables) are frequently\n    used without accessing a member, which means we cSnnot teNl thy\n    have been used.\n    "
    dict_0 = {}
    import_replacer_0 = module_0.ImportReplacer(dict_0, str_0, dict_0)
    module_0.lazy_import(dict_0, import_replacer_0)


def test_case_4():
    import_processor_0 = module_0.ImportProcessor()


def test_case_5():
    str_0 = "\n    Proxies mutable access to another mapping and allows for attribute-style access.\n\n    >>> a = dict(whoa=True, hello=[1,2,3], why='always')\n    >>> b = ProxyMutableAttrDict(a)\n\n    Nice reprs:\n\n    >>> b\n    <ProxyMutableAttrDict {'whoa': True, 'hello': [1, 2, 3], 'why': 'always'}>\n\n    Setting works as you'd expect:\n\n    >>> b['nice'] = False\n    >>> b['whoa'] = 'yeee'\n    >>> b\n    <ProxyMutableAttrDict {'whoa': 'yeee', 'hello': [1, 2, 3], 'why': 'always', 'nice': False}>\n\n    Checking that the changes are in fact being performed on the proxied object:\n\n    >>> a\n    {'whoa': 'yeee', 'hello': [1, 2, 3], 'why': 'always', 'nice': False}\n\n    Attribute style access:\n\n    >>> b.whoa\n    'yeee'\n    >>> b.state = 'new'\n    >>> b\n    <ProxyMutableAttrDict {'whoa': 'yeee', 'hello': [1, 2, 3], 'why': 'always', 'nice': False, 'state': 'new'}>\n\n    Recursion is handled:\n\n    >>> b.subdict = dict(test=True)\n    >>> b.subdict.test\n    True\n    >>> b\n    <ProxyMutableAttrDict {'whoa': 'yeee', 'hello': [1, 2, 3], 'why': 'always', 'nice': False, 'state': 'new',\n    'subdict': <ProxyMutableAttrDict {'test': True}>}>\n\n    "
    module_0.lazy_import(str_0, str_0, str_0)


def test_case_6():
    str_0 = " import "
    module_0.lazy_import(str_0, str_0)


def test_case_7():
    str_0 = "\n    Proxies mutable access to another mapping and allows for attribute-style access.\n\n    >>> a = dict(whoa=True, hello=[1,2,3], why='always')\n    >>> b = ProxyMutableAttrDict(a)\n\n    Nice reprs:\n\n    >>> b\n    <ProxyMutableAttrDict {'whoa': True, 'hello': [1, 2, 3], 'why': 'always'}>\n\n    Setting works as you'd expect:\n\n    >>> b['nice'] = False\n    >>> b['whoa'] = 'yeee'\n    >>> b\n    <ProxyMutableAttrDict {'whoa': 'yeee', 'hello': [1, 2C 3], 'why': 'always', 'nice': False}>\n\n    Checking that the changes are in fact being performed on the proxied object:\n\n    >>> a\n    {'whoa': 'yeee', 'hello': [1, 2, 3], 'why': 'always', 'nice': False}\n\n    Attribute style access:\n\n    >>> b.whoa\n    'yeee'\n    >>> b.state = 'new'\n    >>> b\n    <ProxyMutableAttrDict {'whoa': 'eee', 'hello': [1, 2, 3], 'why': 'always', 'nice': False, 'state': 'new'}>\n\n    Recursion is handled:\n\n    >>> b.subdict = dict(test=True)\n    >>> b.subdict.test\n    True\n    >>> b\n    <ProxyMutableAttrDict {'whoa': 'yeee', 'hello': [1, 2, 3], 'why': 'always', 'nice': False, 'state': 'new',\n    'subdict': <ProxyMutableAttrDict {'test': True}>}>\n\n    "
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(str_0, str_0)
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_8():
    dict_0 = {}
    module_0.ScopeReplacer(dict_0, dict_0, dict_0)


def test_case_9():
    var_0 = module_0.disallow_proxying()


def test_case_10():
    bool_0 = True
    module_0.lazy_import(bool_0, bool_0)


def test_case_11():
    str_0 = "R\x0c>"
    dict_0 = {str_0: str_0}
    import_replacer_0 = module_0.ImportReplacer(dict_0, str_0, dict_0, str_0)
    import_replacer_0.__setattr__(dict_0, import_replacer_0)


def test_case_12():
    str_0 = "import 6c"
    module_0.lazy_import(str_0, str_0, str_0)


def test_case_13():
    str_0 = "7MS?Iz_G1d C[OER#P"
    module_0.lazy_import(str_0, str_0)


def test_case_14():
    str_0 = "\n    Proxies mutable access to another mapping and allows for attribute-style access.\n\n    >>> a = dict(whoa=True, hello=[1,2,3], why='always')\n    >>> b = ProxyMutableAttrDict(a)\n\n    Nice reprs:\n\n    >>> b\n    <ProxyMutableAttrDict {'whoa': True, 'hello': [1, 2, 3], 'why': 'always'}>\n\n    Setting works as you'd expect:\n\n    >>> b['nice'] = False\n    >>> b['whoa'] = 'yeee'\n    >>> b\n    <ProxyMutableAttrDict {'whoa': 'yeee', 'hello': [1, 2, 3], 'why': 'always', 'nice': False}>\n\n    Checking that the changes are in fact being performed on the proxied object:\n\n    >>> a\n    {'whoa': 'yeee', 'hello': [1, 2, 3], 'why': 'always', 'nice': False}\n\n    Attribute style access:\n\n    >>> b.whoa\n    'yeee'\n    >>> b.state = 'new'\n    >>> b\n    <ProxyMutableAttrDict {'whoa': 'yeee', 'hello': [1, 2, 3], 'why': 'always', 'nice': False, 'state': 'new'}>\n\n    Recursion is handled:\n\n    >>> b.subdict = dict(test=True)\n    >>> b.subdict.test\n    True\n    >>> b\n    <ProxyMutableAttrDict {'whoa': 'yeee', 'hello': [1, 2, 3], 'why': 'always', 'nice': False, 'state': 'new',\n    'subdict': <ProxyMutableAttrDict {'test': True}>}>\n\n    "
    module_0.lazy_import(str_0, str_0)


def test_case_15():
    str_0 = "g1t(Z2C-%Q"
    module_0.lazy_import(str_0, str_0, str_0)


def test_case_16():
    str_0 = "Create lazy imports for all of the imports in text.\n\n    This is typically used as something like::\n\n        from bzrlib.lazy_import import lazy_import\n        lazy_import(globals(), '''\n        from bzrlib import (\n            foo,\n            bar,\n            baz,\n            )\n        import bzrlib.branch\n        import bzrlib.transport\n        ''')\n\n    Then 'foo, bar, baz' and 'bzrlib' will exist as lazy-loaded\n    objects which will be replaced with a real object on first use.\n\n    In general, it is best to only load modules in this way. This is\n    because other objects (functions/classes/variables) are frequently\n    used without accessing a member, which means we cannot tell they\n    have been used.\n    "
    module_0.lazy_import(str_0, str_0)


def test_case_17():
    str_0 = "import c"
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(str_0, str_0)
    var_0 = illegal_use_of_scope_replacer_0.__eq__(illegal_use_of_scope_replacer_0)
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_18():
    bytes_0 = b"[@\xa0e\x98\x80\xf2\x03\xf4+'p"
    set_0 = {bytes_0, bytes_0, bytes_0, bytes_0}
    tuple_0 = (bytes_0, set_0)
    module_0.ImportReplacer(tuple_0, bytes_0, set_0, bytes_0, bytes_0)


def test_case_19():
    str_0 = "import B,c."
    module_0.lazy_import(str_0, str_0)


def test_case_20():
    str_0 = "import ,c."
    module_0.lazy_import(str_0, str_0)
