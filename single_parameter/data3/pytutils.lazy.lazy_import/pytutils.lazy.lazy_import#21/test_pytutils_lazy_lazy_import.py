# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pytutils.lazy.lazy_import as module_0
import builtins as module_1


def test_case_0():
    import_processor_0 = module_0.ImportProcessor()
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        import_processor_0, import_processor_0, import_processor_0
    )
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_1():
    bytes_0 = b"\x1f\xf9\x13i~f\x98"
    var_0 = module_0.IllegalUseOfScopeReplacer(bytes_0, bytes_0)


def test_case_2():
    bytes_0 = b"\x84\x1b\xdd\xde\x80a6Rlcf\\,"
    dict_0 = {bytes_0: bytes_0, bytes_0: bytes_0, bytes_0: bytes_0}
    none_type_0 = None
    import_replacer_0 = module_0.ImportReplacer(
        dict_0, none_type_0, none_type_0, bytes_0
    )
    module_0.lazy_import(none_type_0, import_replacer_0)


def test_case_3():
    none_type_0 = None
    float_0 = -1102.136
    module_0.ImportReplacer(none_type_0, float_0, none_type_0)


def test_case_4():
    import_processor_0 = module_0.ImportProcessor()


def test_case_5():
    str_0 = "(F)|/ino?6$z"
    module_0.lazy_import(str_0, str_0, str_0)


def test_case_6():
    import_processor_0 = module_0.disallow_proxying()


def test_case_7():
    base_exception_0 = module_1.BaseException()
    module_0.ImportReplacer(
        base_exception_0,
        base_exception_0,
        base_exception_0,
        base_exception_0,
        base_exception_0,
    )


def test_case_8():
    str_0 = "[RK+ZI'ad:3`1#O<[1<"
    module_0.lazy_import(str_0, str_0, str_0)


def test_case_9():
    str_0 = "Create lazy imports for all of the iports in text.\n\n    This is typically used as something like::\n\n        from bzrlib.lazy_import import lazy_import\n        lazy_import(globals(), '''\n        from bzrlib impot (\n            foo,\n            bar,\n            baz,\n            )\n        import bzrlib.branch\n        import bzrlib.transport\n        ''')\n\n    Then 'foo, bar, baz' and 'bzrlib' will exist as lazy-loaded\n    objects which will be replaced with a real object on first use.\n\n    In general, it is best to only load modules in this way. This is\n    because other objects (functions/classes/variables) are frequently\n    used without accessing a member, which means we cannot tell they\n    have been used.\n    "
    module_0.lazy_import(str_0, str_0)


def test_case_10():
    str_0 = "i\r "
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(str_0, str_0)
    var_0 = illegal_use_of_scope_replacer_0.__eq__(illegal_use_of_scope_replacer_0)
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_11():
    str_0 = "Create lazy imports for all of the imports in text.\n\n    This is typically used as something like::\n\n        from bzrlib.lazy_import import lazy_import\n        lazy_import(globals(), '''\n        from bzrlib import (\n            foo,\n            bar,\n            baz,\n            )\n        import bzrlib.branch\n        import bzrlib.transport\n        ''')\n\n    Then 'foo, bar, baz' and 'bzrlib' will exist as lazy-loaded\n    objects which will be replaced with a real object on first use.\n\n    In general, it is best to only load modules in this way. This is\n    because other objects (functions/classes/variables) are frequently\n    used without accessing a member, which means we cannot tell they\n    have been used.\n    "
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        str_0, str_0, str_0
    )
    var_0 = illegal_use_of_scope_replacer_0.__eq__(str_0)
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_12():
    int_0 = 6
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(int_0, int_0)
    str_0 = "#'#)Vz*:0(l"
    module_0.lazy_import(int_0, str_0)


def test_case_13():
    str_0 = "(|/no?6$zu"
    none_type_0 = None
    module_0.lazy_import(none_type_0, str_0, none_type_0)


def test_case_14():
    bool_0 = True
    dict_0 = {bool_0: bool_0}
    scope_replacer_0 = module_0.ScopeReplacer(dict_0, bool_0, bool_0)
    scope_replacer_0.__getattribute__(bool_0)


def test_case_15():
    str_0 = "^BnvV"
    bool_0 = True
    dict_0 = {str_0: str_0, str_0: str_0, bool_0: bool_0}
    import_replacer_0 = module_0.ImportReplacer(dict_0, str_0, str_0)
    module_0.lazy_import(dict_0, import_replacer_0, str_0)


def test_case_16():
    str_0 = "^BnvV"
    bool_0 = False
    dict_0 = {
        str_0: str_0,
        bool_0: bool_0,
        str_0: str_0,
        str_0: str_0,
        bool_0: bool_0,
        bool_0: bool_0,
    }
    import_replacer_0 = module_0.ImportReplacer(dict_0, bool_0, str_0, dict_0)
    module_0.lazy_import(dict_0, import_replacer_0)
