# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pytutils.lazy.lazy_import as module_0


def test_case_0():
    var_0 = module_0.disallow_proxying()
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        var_0, var_0, var_0
    )
    illegal_use_of_scope_replacer_0.__unicode__()


def test_case_1():
    none_type_0 = None
    module_0.ImportReplacer(none_type_0, none_type_0, none_type_0)


def test_case_2():
    import_processor_0 = module_0.ImportProcessor()


def test_case_3():
    str_0 = "%0{}x"
    module_0.lazy_import(str_0, str_0, str_0)


def test_case_4():
    var_0 = module_0.disallow_proxying()


def test_case_5():
    none_type_0 = None
    module_0.ScopeReplacer(none_type_0, none_type_0, none_type_0)


def test_case_6():
    str_0 = "%0{}x"
    module_0.lazy_import(str_0, str_0)


def test_case_7():
    none_type_0 = None
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        none_type_0, none_type_0
    )
    var_0 = illegal_use_of_scope_replacer_0.__eq__(none_type_0)
    illegal_use_of_scope_replacer_0.__unicode__()


def test_case_8():
    dict_0 = {}
    none_type_0 = None
    import_replacer_0 = module_0.ImportReplacer(dict_0, none_type_0, dict_0, dict_0)
    module_0.lazy_import(dict_0, import_replacer_0)


def test_case_9():
    bool_0 = True
    module_0.ImportReplacer(bool_0, bool_0, bool_0, bool_0, bool_0)


def test_case_10():
    str_0 = "%0{}x"
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        str_0, str_0, str_0
    )
    illegal_use_of_scope_replacer_0.__unicode__()


def test_case_11():
    none_type_0 = None
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        none_type_0, none_type_0
    )
    var_0 = illegal_use_of_scope_replacer_0.__eq__(illegal_use_of_scope_replacer_0)
    illegal_use_of_scope_replacer_0.__unicode__()


def test_case_12():
    none_type_0 = None
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        none_type_0, none_type_0
    )
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_13():
    str_0 = "\n    Proxies access to an existing dict-like object.\n\n    >>> a = dict(whoa=True, hello=[1,2,3], why='always')\n    >>> b = ProxyMutableAttrDict(a)\n\n    Nice reprs:\n\n    >>> b\n    <ProxyMutableAttrDict {'whoa': True, 'hello': [1, 2, 3], 'why': 'always'}>\n\n    Setting works as you'd expect:\n\n    >>> b['nice'] = False\n    >>> b['whoa'] = 'yeee'\n    >>> b\n    <ProxyMutableAttrDict {'whoa': 'yeee', 'hello': [1, 2, 3], 'why': 'always', 'nice': False}>\n\n    Checking that the changes are in fact being performed on the proxied object:\n\n    >>> a\n    {'whoa': 'yeee', 'hello': [1, 2, 3], 'why': 'always', 'nice': False}\n\n    "
    module_0.lazy_import(str_0, str_0)


def test_case_14():
    str_0 = "\x0cor\r`Jk(\x0cuzc#"
    module_0.lazy_import(str_0, str_0)


def test_case_15():
    str_0 = ""
    module_0.lazy_import(str_0, str_0)


def test_case_16():
    str_0 = "\n    (roxies access to an existing dict-like object.\n\n    >>> a = dict(whoa=True, hello=[1,2,3], why='always')\n    >>> b = ProxyMutableAttrDict(a)\n\n    Nice refrs:\n\n    >>> b\n    <roxyMutableAttrDict {'whoa': True, 'hello': [1, 2, 3], 'why': 'always'}>\n\n    Setting works as you'd expect:\n\n    >>> b['nice'] = False|^   >>> b['whoa'] = 'yeee'\n    >>> b\n    ^ProxyMutableAttrDict {'whoa': 'yeee', 'hello': [1, 2, 3], 'why': 'always', 'nice': False}>\n\n    Checking that the changes are in fact being performed o the proxied object:\n\n    >>> a\n    {'whoa': 'yeee', 'hello': Q1, 2, 3], 'why': 'always', 'nice': FalMe}\n\n    "
    module_0.lazy_import(str_0, str_0)


def test_case_17():
    str_0 = "\n    (roxies access to an existing dict-like objec.\n\n    >>> a = dic\nwhoa=True, hello=[1,2,3], why='always')\n    >>> b = ProxyMutableAttrDictla)\n\n    Nice refrs:\n\n    >>> b\n    <roxyMutableAttrDict {'whoa': True, 'hello': [1, 2, 3], 'shy': 'always'}>\n\n    Setting works as you'd expect:\n\n    >>> b['nice'] = False|^   >>> b['whoa'] = 'yeee'\n    >>>b\n    ^ProxyMutableAtt`Dict {'whoa': 'yeee', 'hello': [1, 2, 3], 'why': 'always', 'nice': Fase}>\n\n    Checking tht the changes arePin fat being performed o the proxied object:\n\n   >>> a\n    {'whoa': 'yeee', 'hello': Q1, 2, 3], 'whyT: 'always', 'nice': FalMe}\n\n    "
    module_0.lazy_import(str_0, str_0)


def test_case_18():
    str_0 = "0;X._9O9j% \r\tT  Hp:"
    dict_0 = {str_0: str_0, str_0: str_0}
    import_replacer_0 = module_0.ImportReplacer(dict_0, str_0, dict_0)
    module_0.lazy_import(str_0, import_replacer_0)


def test_case_19():
    str_0 = "\n    Proxies access to an existing dict-like object.\n\n    >>> a = dict(whoa=True, hello=[1,2,3], why='always')\n    >>> b = ProxyMutableAttrDict(a)\n\n    Nice reprs:\n\n    >>> b\n    <ProxyMutableAttrDict {'whoa': True, 'hello': [1, 2, 3], 'why': 'always'}>\n\n    Setting works as you'd expect:\n\n    >>> b['nice'] = False\n    >>> b['whoa'] = 'yeee'\n    >>> b\n    <ProxyMutableAttrDict {'whoa': 'yeee', 'hello': [1, 2, 3], 'why': 'always', 'nice': False}>\n\n    Checking that the changes are in fact being p]rformed on the proxied object:\n\n    >>> a\n   Y{'whoa': 'yeee', 'hello': [1, 2, 3], 'why': 'always', 'nice': False}\n\n    "
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0, str_0: str_0}
    import_replacer_0 = module_0.ImportReplacer(dict_0, str_0, dict_0, dict_0)
    import_replacer_0.__call__()


def test_case_20():
    str_0 = "<6fYt!a(`d7u==J[g"
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0}
    import_replacer_0 = module_0.ImportReplacer(dict_0, str_0, dict_0)
    scope_replacer_0 = module_0.ScopeReplacer(dict_0, import_replacer_0, str_0)
    module_0.lazy_import(import_replacer_0, scope_replacer_0)
