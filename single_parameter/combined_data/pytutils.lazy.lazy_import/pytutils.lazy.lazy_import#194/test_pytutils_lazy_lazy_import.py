# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pytutils.lazy.lazy_import as module_0


def test_case_0():
    bool_0 = True
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        bool_0, bool_0, bool_0
    )
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_1():
    bool_0 = False
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        bool_0, bool_0, bool_0
    )


def test_case_2():
    bool_0 = False
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        bool_0, bool_0, bool_0
    )
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_3():
    none_type_0 = None
    module_0.ImportReplacer(none_type_0, none_type_0, none_type_0, none_type_0)


def test_case_4():
    import_processor_0 = module_0.ImportProcessor()


def test_case_5():
    set_0 = set()
    import_processor_0 = module_0.ImportProcessor(set_0)


def test_case_6():
    str_0 = "Create lazy imports for all of the imports in text.\n\n    This is typically used as something like::\n\n        from bzrlib.lazy_import import lazy_import\n        lazy_import(globals(), '''\n        from bzrlib import (\n            foo,\n            bar,\n            baz,\n            )\n        import bzrlib.branch\n        import bzrlib.transport\n        ''')\n\n    Then 'foo, bar, baz' and 'bzrlib' will exist as lazy-loaded\n    objects which will be replaced with a real object on first use.\n\n    In general, it is best to only load odules in this way. This is\n    because other objects (functions/classs/variables) are frequently\n    used withcut accessing a member, which means we cannot tell they\n    have been used.\n    "
    module_0.lazy_import(str_0, str_0, str_0)


def test_case_7():
    var_0 = module_0.disallow_proxying()


def test_case_8():
    none_type_0 = None
    module_0.lazy_import(none_type_0, none_type_0)


def test_case_9():
    var_0 = module_0.disallow_proxying()
    var_1 = var_0.__str__()
    dict_0 = {}
    import_replacer_0 = module_0.ImportReplacer(dict_0, var_1, dict_0, var_1, dict_0)
    import_replacer_0.__call__()


def test_case_10():
    str_0 = "Create lazy imports for all of the imports in text.\n\n    This is typically used as something like::\n\n        from bzrlib.lazy_import import lazy_import\n        lazy_import(globals(), '''\n        from bzrlib import (\n            foo,\n            bar,\n            baz,\n            )\n        import bzrlib.branch\n        import bzrlib.transport\n        ''')\n\n    Then 'foo, bar, baz' and 'bzrlib' will exist as lazy-loaded\n    objects which will be replaced with a real object on first use.\n\n    In general, it is best to only load modules in this way. This is\n    because other objects (functions/classes/variables) are frequently\n    used without accessing a member, which means we cannot tell they\n    have been used.\n    "
    var_0 = module_0.IllegalUseOfScopeReplacer(str_0, str_0, str_0)
    var_1 = var_0.__eq__(var_0)
    var_0.__unicode__()


def test_case_11():
    int_0 = 2152
    str_0 = ""
    module_0.lazy_import(int_0, str_0)


def test_case_12():
    int_0 = 2185
    str_0 = "lk|P"
    module_0.ImportReplacer(str_0, int_0, str_0, int_0, int_0)


def test_case_13():
    int_0 = 2179
    str_0 = "}|("
    module_0.lazy_import(int_0, str_0)


def test_case_14():
    bool_0 = False
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        bool_0, bool_0, bool_0
    )
    var_0 = illegal_use_of_scope_replacer_0.__eq__(bool_0)
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_15():
    str_0 = ";5^'y#EL"
    module_0.lazy_import(str_0, str_0, str_0)


def test_case_16():
    var_0 = module_0.disallow_proxying()
    dict_0 = {}
    import_replacer_0 = module_0.ImportReplacer(dict_0, var_0, var_0)
    import_replacer_0.__repr__()


def test_case_17():
    var_0 = module_0.disallow_proxying()
    var_1 = var_0.__repr__()
    dict_0 = {}
    import_replacer_0 = module_0.ImportReplacer(dict_0, var_0, var_1)
    module_0.lazy_import(var_0, import_replacer_0)
