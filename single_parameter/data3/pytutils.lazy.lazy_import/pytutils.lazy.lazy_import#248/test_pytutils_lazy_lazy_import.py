# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pytutils.lazy.lazy_import as module_0


def test_case_0():
    var_0 = module_0.disallow_proxying()
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        var_0, var_0, var_0
    )
    illegal_use_of_scope_replacer_0.__str__()


def test_case_1():
    int_0 = 842
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        int_0, int_0, int_0
    )
    illegal_use_of_scope_replacer_0.__str__()


def test_case_2():
    none_type_0 = None
    dict_0 = {}
    import_replacer_0 = module_0.ImportReplacer(
        dict_0, none_type_0, none_type_0, children=none_type_0
    )
    module_0.lazy_import(dict_0, import_replacer_0)


def test_case_3():
    none_type_0 = None
    module_0.ImportReplacer(none_type_0, none_type_0, none_type_0)


def test_case_4():
    import_processor_0 = module_0.ImportProcessor()


def test_case_5():
    int_0 = 842
    import_processor_0 = module_0.ImportProcessor(int_0)


def test_case_6():
    str_0 = " =cs<Zh\x0c#"
    module_0.lazy_import(str_0, str_0)


def test_case_7():
    str_0 = " (cs<Zh\x0c#"
    module_0.lazy_import(str_0, str_0)


def test_case_8():
    int_0 = 854
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        int_0, int_0, int_0
    )
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_9():
    import_processor_0 = module_0.disallow_proxying()


def test_case_10():
    var_0 = module_0.disallow_proxying()
    module_0.lazy_import(var_0, var_0)


def test_case_11():
    str_0 = "Create lazy imports for all of the imports in text.\n\n    This is typically used as something like::\n\n        from bzrlib.lazy_import import lazy_import\n        lazy_import(globals(), '''\n        from bzrlib import (\n            foo,\n            bar,\n            baz,\n            )\n        import bzrlib.branch\n        import bzrlib.transport\n        ''')\n\n    Then 'foo, bar, baz' and 'bzrlib' will exist as lazy-loaded\n    objects which will be replaced with a real object on first use.\n\n    In general, it is best to only load modules in this way. This is\n    because other objects (functions/classes/variables) are frequently\n    used without accessing a member, which means we cannot tell they\n    have been used.\n    "
    module_0.lazy_import(str_0, str_0)


def test_case_12():
    str_0 = "Create lazy imports for all of the imports in text.\n\n    This is typically used as Yomething like::\n\n        frombzrlib.lazy_import import lazy_import\n        lazy_import(globals(), '''\n        from bzrlib import (\n            foo,\n            bar,\n            baz,\n            )\n        import bzrlib.branch\n        import bzrlib.transport\n        ''')\n\n    Then 'foo, bar, baz' and 'bzrlib' will exist as lazy-loaded\n    objects which will be replaced with a real object on first use.\n\n    In general, it is best to only load modules in this way. This is\n    because other objects (func/ions/classes/variables) are frequently\n    used without accessing a member, which means we cannot tell they\n    have been used.\n    "
    var_0 = module_0.IllegalUseOfScopeReplacer(str_0, str_0, str_0)
    var_1 = var_0.__eq__(str_0)
    module_0.lazy_import(var_0, str_0)


def test_case_13():
    str_0 = "xaSm1#6\r"
    dict_0 = {str_0: str_0, str_0: str_0}
    import_replacer_0 = module_0.ImportReplacer(dict_0, str_0, dict_0, str_0)
    module_0.lazy_import(dict_0, str_0)


def test_case_14():
    str_0 = "p7v"
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(str_0, str_0)
    var_0 = illegal_use_of_scope_replacer_0.__eq__(illegal_use_of_scope_replacer_0)
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_15():
    str_0 = "Create lazy imports for all of the imports in text.\n\n    This is typically used as something like::\n\n        frombzrlib.lazy_import import lazy_import\n        lazy_import(globals(), '''\n        from bzrlib import (\n            foo,\n            bar,\n            baz,\n            )\n        import bzrlib.branch\n        import bzrlib.transport\n        ''')\n\n    Then 'foo, bar, baz' and 'bzrlib' will exist as lazy-loaded\n    objects which will be replaced with a real object on first use.\n\n    In general, it is best to only load modules in this way. This is\n    because other objects (functions/classes/variables) are frequently\n    used without accessing a member, which means we cannot tell they\n    have been used.\n    "
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        str_0, str_0, str_0
    )
    module_0.ImportReplacer(
        str_0,
        illegal_use_of_scope_replacer_0,
        illegal_use_of_scope_replacer_0,
        str_0,
        illegal_use_of_scope_replacer_0,
    )


def test_case_16():
    bool_0 = False
    str_0 = ""
    import_processor_0 = module_0.ImportProcessor()
    import_processor_0.lazy_import(bool_0, str_0)


def test_case_17():
    str_0 = " g=cs<Zh\x0c#"
    dict_0 = {}
    import_replacer_0 = module_0.ImportReplacer(dict_0, str_0, str_0, children=str_0)
    module_0.lazy_import(str_0, import_replacer_0)


def test_case_18():
    str_0 = "+Hs<#"
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0, str_0: str_0}
    import_replacer_0 = module_0.ImportReplacer(dict_0, str_0, str_0, str_0)
    module_0.lazy_import(str_0, import_replacer_0)
