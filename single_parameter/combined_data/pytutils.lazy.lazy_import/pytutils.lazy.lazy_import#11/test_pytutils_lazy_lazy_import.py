# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pytutils.lazy.lazy_import as module_0
import builtins as module_1


def test_case_0():
    str_0 = "_real_obj"
    var_0 = module_0.IllegalUseOfScopeReplacer(str_0, str_0, str_0)
    var_0.__str__()


def test_case_1():
    none_type_0 = None
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        none_type_0, none_type_0
    )


def test_case_2():
    none_type_0 = None
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        none_type_0, none_type_0, none_type_0
    )
    illegal_use_of_scope_replacer_0.__str__()


def test_case_3():
    str_0 = "\n    Update and/or insert query parameters in a URL.\n\n    >>> update_query_params('http://example.com?foo=bar&biz=baz', dict(foo='stuff'))\n    'http://example.com?...foo=stuff...'\n\n    :param url: URL\n    :type url: str\n    :param kwargs: Query parameters\n    :type kwargs: dict\n    :return: Modified URL\n    :rtype: str\n    "
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(str_0, str_0)
    var_0 = illegal_use_of_scope_replacer_0.__eq__(str_0)
    illegal_use_of_scope_replacer_0.__str__()


def test_case_4():
    str_0 = "\n    Derive a namespace from the module containing the caller's caller.\n\n    :return: the fully qualified python name of a module.\n    :rtype: str\n    "
    none_type_0 = None
    module_0.ImportReplacer(none_type_0, none_type_0, str_0, str_0)


def test_case_5():
    none_type_0 = None
    module_0.ImportReplacer(
        none_type_0, none_type_0, none_type_0, none_type_0, none_type_0
    )


def test_case_6():
    import_processor_0 = module_0.ImportProcessor()


def test_case_7():
    str_0 = "\n    Derive a namespace from the module containing the caller's caller.\n\n    :return: the fully qualified python name of a module.\n    :rtype: str\n    "
    module_0.lazy_import(str_0, str_0, str_0)


def test_case_8():
    str_0 = "\n    Derive a namespace from the module containing the caller's 'aller.\n\n    :return: the fully-qualified python name of a module.\n    :rtype: str\n    "
    module_0.lazy_import(str_0, str_0)


def test_case_9():
    var_0 = module_0.disallow_proxying()


def test_case_10():
    str_0 = "\n    Derive a namespace from the module containing the caller's caller.\n\n    :return: the fully qualified python name of a module.\n    :rtype: str\n    "
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(str_0, str_0)
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_11():
    bool_0 = True
    module_0.lazy_import(bool_0, bool_0)


def test_case_12():
    import_processor_0 = module_0.ImportProcessor()
    str_0 = "\r"
    module_0.lazy_import(str_0, str_0, str_0)


def test_case_13():
    str_0 = "|u:8c;#nleoI)"
    module_0.lazy_import(str_0, str_0)


def test_case_14():
    str_0 = "o\t,3|Yr_/]J9l(X\tQ ="
    module_0.lazy_import(str_0, str_0)


def test_case_15():
    str_0 = "Create a temporary object in the specified scope.\n        Once used, a real object will be placed in the scope.\n\n        :param scope: The scope the object should appear in\n        :param factory: A callable that will create the real object.\n            It will be passed (self, scope, name)\n        :param name: The variable name in the given scope.\n        "
    module_0.lazy_import(str_0, str_0)


def test_case_16():
    base_exception_0 = module_1.BaseException()
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        base_exception_0, base_exception_0
    )
    var_0 = illegal_use_of_scope_replacer_0.__eq__(illegal_use_of_scope_replacer_0)
    illegal_use_of_scope_replacer_0.__unicode__()


def test_case_17():
    str_0 = "Create a temporary object in the specified scope.\n        Once used, a real object will be placed in the scope.\n\n        :param scope: The scope the object should appear in\n        :param factory: A callable that will create the real object.\n            It will be passed (self, scope, name)\n        :param name: The variable name in the given scope.\n        "
    dict_0 = {str_0: str_0}
    none_type_0 = None
    scope_replacer_0 = module_0.ScopeReplacer(dict_0, str_0, none_type_0)
    module_0.lazy_import(str_0, scope_replacer_0)


def test_case_18():
    complex_0 = 919 + 3209.7101j
    none_type_0 = None
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        complex_0, none_type_0
    )
    float_0 = -1398.0
    module_0.ImportReplacer(float_0, float_0, float_0, float_0, float_0)


def test_case_19():
    str_0 = "Create a temporary object in the spec(fied scpe.\n        Once used, a real object will be placed in the scope.\n\n       :para scope: The scope the object should appear in\n        :param factory: A callable that will create the real object.\n            It will be pased (self, scope, name)\n        :param na: The variable name in the given scope.\n        "
    module_0.lazy_import(str_0, str_0)
