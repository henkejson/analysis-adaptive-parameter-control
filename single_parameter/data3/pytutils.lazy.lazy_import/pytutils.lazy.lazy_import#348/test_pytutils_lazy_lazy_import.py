# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pytutils.lazy.lazy_import as module_0


def test_case_0():
    str_0 = "d5NUs^8=s3O4d]R"
    var_0 = module_0.IllegalUseOfScopeReplacer(str_0, str_0, str_0)
    var_0.__repr__()


def test_case_1():
    none_type_0 = None
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        none_type_0, none_type_0
    )
    illegal_use_of_scope_replacer_0.__str__()


def test_case_2():
    bool_0 = True
    dict_0 = {bool_0: bool_0, bool_0: bool_0, bool_0: bool_0, bool_0: bool_0}
    import_replacer_0 = module_0.ImportReplacer(dict_0, bool_0, bool_0, bool_0)
    module_0.lazy_import(import_replacer_0, import_replacer_0)


def test_case_3():
    str_0 = '"}}:!,m'
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0}
    import_replacer_0 = module_0.ImportReplacer(dict_0, str_0, dict_0)
    module_0.lazy_import(import_replacer_0, import_replacer_0)


def test_case_4():
    import_processor_0 = module_0.ImportProcessor()


def test_case_5():
    str_0 = "CH%No4-yMi"
    module_0.lazy_import(str_0, str_0, str_0)


def test_case_6():
    var_0 = module_0.disallow_proxying()


def test_case_7():
    str_0 = "d5NUs^8=s3O4d]R"
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(str_0, str_0)
    var_0 = illegal_use_of_scope_replacer_0.__eq__(str_0)
    module_0.lazy_import(str_0, str_0)


def test_case_8():
    str_0 = ""
    module_0.lazy_import(str_0, str_0)


def test_case_9():
    str_0 = "\n    Provides a basic per-process mapping container that wipes itself if the current PID changed since the last get/set.\n\n    Aka `threading.local()`, but for processes instead of threads.\n\n    >>> plocal = ProcessLocal()\n    >>> plocal['test'] = True\n    >>> plocal['test']\n    True\n    >>> plocal._handle_pid(new_pid=-1)  # Emulate a PID change by forcing it to be something invalid.\n    >>> plocal['test']                  # Mapping wipes itself since PID is different than what's stored.\n    Traceback (most recent call last):\n        ...\n    KeyError: ...\n\n    "
    module_0.lazy_import(str_0, str_0)


def test_case_10():
    str_0 = "S(D\n<?%O)Se"
    module_0.lazy_import(str_0, str_0)


def test_case_11():
    bytes_0 = b"\xbf\x01\xb1qv]\x17G\xb0\x7f\xa8\x1d?l"
    module_0.ImportReplacer(bytes_0, bytes_0, bytes_0, bytes_0, bytes_0)


def test_case_12():
    none_type_0 = None
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        none_type_0, none_type_0
    )
    var_0 = illegal_use_of_scope_replacer_0.__eq__(illegal_use_of_scope_replacer_0)
    illegal_use_of_scope_replacer_0.__str__()


def test_case_13():
    dict_0 = {}
    bool_0 = True
    scope_replacer_0 = module_0.ScopeReplacer(dict_0, dict_0, bool_0)
    var_0 = module_0.ScopeReplacer(dict_0, scope_replacer_0, bool_0)
    var_0.__call__(**dict_0)


def test_case_14():
    str_0 = "S(0\n<%)"
    module_0.lazy_import(str_0, str_0)


def test_case_15():
    str_0 = '"}:!,m'
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0}
    none_type_0 = None
    import_replacer_0 = module_0.ImportReplacer(dict_0, none_type_0, str_0, dict_0)
    import_replacer_0.__getattribute__(dict_0)
