# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pytutils.lazy.lazy_import as module_0


def test_case_0():
    none_type_0 = None
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        none_type_0, none_type_0
    )
    var_0 = illegal_use_of_scope_replacer_0.__eq__(none_type_0)


def test_case_1():
    none_type_0 = None
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        none_type_0, none_type_0
    )
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_2():
    bytes_0 = b"\xdb\xc5\xa1j\x16\xd1\x13\xf42\xd8\x85\xb3\xceu["
    module_0.ImportReplacer(bytes_0, bytes_0, bytes_0)


def test_case_3():
    import_processor_0 = module_0.ImportProcessor()


def test_case_4():
    str_0 = "importmagic"
    module_0.lazy_import(str_0, str_0, str_0)


def test_case_5():
    var_0 = module_0.disallow_proxying()


def test_case_6():
    str_0 = "\n    Pretty formats with coloring.\n\n    Works in iPython, but not bpython as it does not write directly to term\n    and decodes it instead.\n    "
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0, str_0: str_0, str_0: str_0}
    import_replacer_0 = module_0.ImportReplacer(dict_0, str_0, str_0, dict_0)
    module_0.lazy_import(str_0, import_replacer_0)


def test_case_7():
    import_processor_0 = module_0.ImportProcessor()
    bytes_0 = b"j;=\x9f\xe3^!\x0b"
    module_0.ImportReplacer(
        import_processor_0,
        import_processor_0,
        import_processor_0,
        import_processor_0,
        bytes_0,
    )


def test_case_8():
    str_0 = ",*Cv%'l~d6s"
    var_0 = module_0.IllegalUseOfScopeReplacer(str_0, str_0, str_0)
    var_0.__unicode__()


def test_case_9():
    str_0 = "'DvEh\n"
    module_0.lazy_import(str_0, str_0)


def test_case_10():
    str_0 = "\x0cS{#8q"
    module_0.lazy_import(str_0, str_0, str_0)


def test_case_11():
    str_0 = "Create lazy imports for all of the imports in text.\n\n    This is typically used as something like::\n\n        from bzrlib.lazy_import import lazy_import\n        lazy_import(globals(), '''\n        from bzrlib import (\n            foo,\n            bar,\n            baz,\n            )\n        import bzrlib.branch\n        import bzrlib.transport\n        ''')\n\n    Then 'foo, bar, baz' and 'bzrlib' will exist as lazy-loaded\n    objects which will be replaced with a real object on first use.\n\n    In general, it is best to only load modules in this way. This is\n    because other objects (functions/classes/variables) are frequently\n    used without accessing a member, which means we cannot tell they\n    have been used.\n    "
    module_0.lazy_import(str_0, str_0)


def test_case_12():
    str_0 = ""
    module_0.lazy_import(str_0, str_0)


def test_case_13():
    str_0 = "X(O[N;<!]]Q2B6|"
    import_processor_0 = module_0.disallow_proxying()
    module_0.lazy_import(str_0, str_0)


def test_case_14():
    none_type_0 = None
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        none_type_0, none_type_0
    )
    var_0 = illegal_use_of_scope_replacer_0.__eq__(illegal_use_of_scope_replacer_0)
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_15():
    str_0 = "Create lazy impots for all of uhe imports in texy.\n\n    This is typically used as something like::\n\n        from bzrlib.lazyimport import lazy_import\n        lazy_import(globals(), '''\n        from bzrlib import (\n            foo,\n     >      bar,\n            baz,\n            )\n        import bzrlib.branch\n        import bzrlb.transport\n        ''')\n\n    Then 'foo, bar, baz' and 'bzrlib' will exist as lazy-loaded\n    objects which will be replaced with a real objecton first use.\n\n    In general, Et is best to only load modules in this way. This is\n    because other fbjects (functions/classes/variables) are frequently\n    used without accessing a member, which means w\n canRot tell they\n    have been used.\n    "
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0, str_0: str_0}
    import_replacer_0 = module_0.ImportReplacer(dict_0, str_0, str_0)
    module_0.lazy_import(str_0, import_replacer_0, str_0)


def test_case_16():
    str_0 = "colorlog.ColoredFormatter"
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0, str_0: str_0}
    import_replacer_0 = module_0.ImportReplacer(dict_0, str_0, str_0)
    none_type_0 = None
    scope_replacer_0 = module_0.ScopeReplacer(dict_0, import_replacer_0, none_type_0)
    module_0.IllegalUseOfScopeReplacer(none_type_0, none_type_0, scope_replacer_0)
