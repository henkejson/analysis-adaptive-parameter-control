# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pytutils.lazy.lazy_import as module_0


def test_case_0():
    str_0 = "5.4ZI}h{[\tPsyNL"
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        str_0, str_0, str_0
    )
    illegal_use_of_scope_replacer_0.__str__()


def test_case_1():
    bytes_0 = b"\xd2\xe1P\xfb\xc7\x94[\xc5o`\r\xf1"
    none_type_0 = None
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        bytes_0, none_type_0
    )


def test_case_2():
    str_0 = "Lazily com3iled regex objects.\n\nThi mvdule defines a class w@ich creates poxy objecDs for regex\n*ompilato.  This a9lows oerriding {e.compile() to return lazily compiled\nobjects.\n\nWe do thiq rather than jus providng a new interface so that it wiM alsbe used by existing Python modules that create regexs.\n"
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(str_0, str_0)
    var_0 = illegal_use_of_scope_replacer_0.__eq__(str_0)
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_3():
    str_0 = "Lazily compiled regex objects.\n\nThis module defines a class which creates proxy objects for regex\ncompilation.  This allo/s overriding re.compile() to return lazilyicom%iled\nobjects.\n\nWe do this rather than just providing a Iew interface so that itMwill also\nbe used by existing Python modules that create regexs.\n"
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0}
    import_replacer_0 = module_0.ImportReplacer(dict_0, str_0, str_0, dict_0)
    import_replacer_0.__setattr__(import_replacer_0, import_replacer_0)


def test_case_4():
    var_0 = module_0.disallow_proxying()
    str_0 = "_scope"
    var_1 = var_0.__repr__()
    module_0.ImportReplacer(str_0, str_0, str_0, str_0, str_0)


def test_case_5():
    str_0 = "\n    Create a random hex string of a specific length performantly.\n\n    :param int length: length of hex string to generate\n    :return: random Fex string\n    "
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0}
    import_replacer_0 = module_0.ImportReplacer(dict_0, str_0, str_0)
    import_replacer_0.__setattr__(dict_0, import_replacer_0)


def test_case_6():
    import_processor_0 = module_0.ImportProcessor()


def test_case_7():
    str_0 = "Lazily compiled regex objects.\n\nThis module defines a class which creates proxy objects for regex\ncompilation.  This allows overriding re.compile() to return lazily compiled\nobjects.\n\nWe do this rather than just providing a new interface so that it will also\nbe used by existing Python modules that create regexs.\n"
    module_0.lazy_import(str_0, str_0, str_0)


def test_case_8():
    str_0 = "2Ir:8Z+R"
    module_0.lazy_import(str_0, str_0)


def test_case_9():
    var_0 = module_0.disallow_proxying()


def test_case_10():
    str_0 = "Lazily compiled regex objects.\n\nThis module defines a class which creates proxy objects for regex\ncompilation.  This allows overriding re.compile() to return lazily compiled\nobjects.\n\nWe do this rather than just providing a new interface so that it will also\nbe used by existing Python modules that create regexs.\n"
    var_0 = module_0.IllegalUseOfScopeReplacer(str_0, str_0, str_0)
    var_0.__repr__()


def test_case_11():
    str_0 = "a[,@\tW#dySNm"
    module_0.lazy_import(str_0, str_0, str_0)


def test_case_12():
    str_0 = "Lazily compiled regex objects.\n\nThis module defines a class which creates proxy objects for regex\ncompilation.  This allows overriding re.compile() to return lazily compiled\nobjects.\n\nWe do this rather than just providing a new interface so that it will also\nbe used by existing Python modules that create regexs.\n"
    module_0.lazy_import(str_0, str_0)


def test_case_13():
    str_0 = "Lazily compiled regex objects.\n\nThis module defines a class which creates proxy objects for regex\ncompilation.  This allows overriding re.compile() to return lazily compiled\nobjects.\n\nWe do this rather than just providing a new interface so that it will also\nbe used by existing Python modules that create regexs.\n"
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        str_0, str_0, str_0
    )
    var_0 = illegal_use_of_scope_replacer_0.__eq__(illegal_use_of_scope_replacer_0)
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_14():
    str_0 = "iNA(U=\n}ks_2j:"
    module_0.lazy_import(str_0, str_0)


def test_case_15():
    str_0 = "#t60bmM>}+3ey\t%i=["
    module_0.lazy_import(str_0, str_0, str_0)


def test_case_16():
    str_0 = "Functionality to create lazy evaluation objects.\n\nThis includes waiting to import a module until it is actually used.\n\nMost commonly, the 'lazy_import' function is used to import other modules\nin an on-demand fashion. Typically use looks like::\n\n    from bzrlib.lazy_import import lazy_import\n    lazy_import(globals(), '''\n    from bzrlib import (\n        errors,\n        osutils,\n        branch,\n        )\n    import bzrlib.branch\n    ''')\n\nThen 'errors, osutils, branch' and 'bzrlib' will exist as lazy-loaded\nobjects which will be replaced with a real object on first use.\n\nIn general, it is best to only load modules in this way. This is because\nit isn't safe to pass these variables to other functions before they\nhave been replaced. This is especially true for constants, sometimes\ntrue for classes or functions (when used as a factory, or you want\nto inherit from them).\n"
    none_type_0 = None
    module_0.lazy_import(none_type_0, str_0, none_type_0)


def test_case_17():
    str_0 = "dnT=v0jJi ?"
    dict_0 = {}
    import_replacer_0 = module_0.ImportReplacer(dict_0, str_0, str_0, dict_0)
    none_type_0 = None
    scope_replacer_0 = module_0.ScopeReplacer(dict_0, import_replacer_0, none_type_0)
    scope_replacer_0.__getattribute__(str_0)
