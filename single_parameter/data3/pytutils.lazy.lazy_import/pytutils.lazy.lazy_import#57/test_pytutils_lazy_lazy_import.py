# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pytutils.lazy.lazy_import as module_0


def test_case_0():
    str_0 = '" '
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(str_0, str_0)
    module_0.lazy_import(illegal_use_of_scope_replacer_0, str_0)


def test_case_1():
    str_0 = '" '
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(str_0, str_0)
    illegal_use_of_scope_replacer_0.__str__()


def test_case_2():
    str_0 = "Restore the original function to re.compile().\n\n    It is safe to call reset_compile() multiple times, it will always\n    restore re.compile() to the value that existed at import time.\n    Though the first call will reset back to the original (it doesn't\n    track nestig level)\n    "
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(str_0, str_0)
    var_0 = illegal_use_of_scope_replacer_0.__eq__(str_0)
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_3():
    str_0 = "_impopt_replacer_chi5drn"
    dict_0 = {
        str_0: str_0,
        str_0: str_0,
        str_0: str_0,
        str_0: str_0,
        str_0: str_0,
        str_0: str_0,
        str_0: str_0,
    }
    import_replacer_0 = module_0.ImportReplacer(dict_0, str_0, str_0, str_0)
    module_0.lazy_import(import_replacer_0, import_replacer_0, str_0)


def test_case_4():
    var_0 = module_0.disallow_proxying()
    module_0.ImportReplacer(var_0, var_0, var_0)


def test_case_5():
    import_processor_0 = module_0.ImportProcessor()


def test_case_6():
    str_0 = '" '
    module_0.lazy_import(str_0, str_0, str_0)


def test_case_7():
    str_0 = '" '
    module_0.lazy_import(str_0, str_0)


def test_case_8():
    str_0 = "("
    module_0.lazy_import(str_0, str_0)


def test_case_9():
    var_0 = module_0.disallow_proxying()


def test_case_10():
    bool_0 = False
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(bool_0, bool_0)
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_11():
    bytes_0 = b"P\xb9\xd2\xe04l\x9c"
    var_0 = module_0.disallow_proxying()
    var_1 = var_0.__str__()
    list_0 = [bytes_0, bytes_0]
    module_0.ImportReplacer(list_0, list_0, bytes_0, list_0, bytes_0)


def test_case_12():
    str_0 = "Restore the original function to re.compile().\n\n    It is safe to call reset_compile() multiple times, it will always\n    restore re.compile() to the value that existed at import time.\n    Though the first call will reset back to the original (it doesn't\n    track nesting level)\n    "
    module_0.lazy_import(str_0, str_0, str_0)


def test_case_13():
    str_0 = "This converts a 'from foo import*bar' string into an import map.\n\n        :param from_str: The import string to process\n        "
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        str_0, str_0, str_0
    )
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_14():
    str_0 = "Restore the original function to re.compile().\n\n    It is safe to call reset_compile() multiple times, it will always\n    restore re.compile() to the value that existed at import time.\n    Though the first call will reset back to the original (it doesn't\n    track nestig level)\n    "
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(str_0, str_0)
    var_0 = illegal_use_of_scope_replacer_0.__eq__(illegal_use_of_scope_replacer_0)
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_15():
    str_0 = ""
    var_0 = module_0.disallow_proxying()
    module_0.lazy_import(str_0, str_0)


def test_case_16():
    str_0 = "This converts a 'from foo import bar' string into an import map.\n\n        :param from_str: The import string to process\n        "
    module_0.lazy_import(str_0, str_0, str_0)


def test_case_17():
    str_0 = '2u(BJ1u":h?\x0c'
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0}
    import_replacer_0 = module_0.ImportReplacer(dict_0, str_0, str_0)
    module_0.lazy_import(str_0, import_replacer_0)


def test_case_18():
    str_0 = "T[D.^pz@OD'6ODqZ#Ax"
    module_0.lazy_import(str_0, str_0, str_0)


def test_case_19():
    str_0 = "Functionality to create lazy evaluation objects.\n\nThis includes waiting to import a module until it is actually used.\n\nMost commonly, the 'lazy_import' function is used to import other modules\nin an on-demand fashion. Typically use looks like::\n\n    from bzrlib.lazy_import import lazy_import\n    lazy_import(globals(), '''\n    from bzrlib import (\n        errors,\n        osutils,\n        branch,\n        )\n    import bzrlib.branch\n    ''')\n\nThen 'errors, osutils, branch' and 'bzrlib' will exist as lazy-loaded\nobjects which will be replaced with a real object on first use.\n\nIn general, it is best to only load modules in this way. This is because\nit isn't safe to pass these variables to other functions before they\nhave been replaced. This is especially true for constants, sometimes\ntrue for classes or functions (when used as a factory, or you want\nto inherit from them).\n"
    module_0.lazy_import(str_0, str_0)
