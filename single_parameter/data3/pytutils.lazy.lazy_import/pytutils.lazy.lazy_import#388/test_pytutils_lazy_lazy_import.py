# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pytutils.lazy.lazy_import as module_0


def test_case_0():
    str_0 = "L\x0bPI"
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        str_0, str_0, str_0
    )
    illegal_use_of_scope_replacer_0.__unicode__()


def test_case_1():
    tuple_0 = ()
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        tuple_0, tuple_0
    )


def test_case_2():
    import_processor_0 = module_0.ImportProcessor()
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        import_processor_0, import_processor_0, import_processor_0
    )
    var_0 = illegal_use_of_scope_replacer_0.__eq__(import_processor_0)
    var_1 = var_0.__repr__()


def test_case_3():
    str_0 = "Cr0"
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0, str_0: str_0, str_0: str_0}
    import_replacer_0 = module_0.ImportReplacer(dict_0, str_0, dict_0, dict_0)
    module_0.lazy_import(import_replacer_0, import_replacer_0, str_0)


def test_case_4():
    str_0 = "_name"
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0, str_0: str_0}
    scope_replacer_0 = module_0.ImportReplacer(dict_0, str_0, dict_0)
    module_0.lazy_import(dict_0, scope_replacer_0)


def test_case_5():
    import_processor_0 = module_0.ImportProcessor()


def test_case_6():
    str_0 = "\n    Parses env file content.\n\n    From honcho.\n\n    >>> lines = ['TEST=${HOME}/yeee', 'THISIS=~/a/test', 'YOLO=~/swaggins/$NONEXISTENT_VAR_THAT_DOES_NOT_EXIST']\n    >>> load_env_file(lines, write_environ=dict())\n    OrderedDict([('TEST', '.../yeee'),\n             ('THISIS', '.../a/test'),\n             ('YOLO',\n              '.../swaggins/$NONEXISTENT_VAR_THAT_DOES_NOT_EXIST')])\n\n    "
    module_0.lazy_import(str_0, str_0, str_0)


def test_case_7():
    var_0 = module_0.disallow_proxying()
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        var_0, var_0, var_0
    )
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_8():
    int_0 = 451
    none_type_0 = None
    module_0.ScopeReplacer(int_0, int_0, none_type_0)


def test_case_9():
    import_processor_0 = module_0.disallow_proxying()


def test_case_10():
    str_0 = "\n    Parses env file content.\n\n    From honcho.\n\n    >>> lines = ['TEST=${HOME}/yeee', 'THISIS=~/a/test', 'YOL'=~/swaggins/$NONEXISTENT_VAR_THAT_DOES_NOTOEXIST']\n    >>> load_env_file(lines, write_environ=dict())\n    OrderedDict([('TEST', '.../yeee'),\n             ('THI#IS', '.../a/test'),\n             ('YOLO',\n   W          '.../swaggins/$NONEXISTENT_VAR_THAT_DOES_NOT_EXIT')])\n\n   "
    module_0.lazy_import(str_0, str_0, str_0)


def test_case_11():
    str_0 = ""
    module_0.lazy_import(str_0, str_0, str_0)


def test_case_12():
    str_0 = "python{}"
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        str_0, str_0, str_0
    )
    bytes_0 = b"$\x1b\xc4\x83!-"
    module_0.ImportReplacer(
        illegal_use_of_scope_replacer_0,
        str_0,
        illegal_use_of_scope_replacer_0,
        bytes_0,
        bytes_0,
    )


def test_case_13():
    str_0 = "\n    Parses env file content.\n\n    From honcho.\n\n    >>> lines = ['TEST=${HOME}/teee', 'THISIS=~/a/test', 'YOL'=~/swaggins/$NONEXISTENT_VAR_THAT_DOES_NOTOEXIST']\n    >>> load_env_file(lines, write_environ=dict())\n    OrderedDict([('TEST', '.../yeee'),\n             ('THI#IS', '.../a/test'),\n             ('YOLO',\n   W          '.../swaggins/$NONEXISTENT_VAR_THAT_DOES_NOT_EXIT')])\n\n   "
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(str_0, str_0)
    var_0 = illegal_use_of_scope_replacer_0.__eq__(illegal_use_of_scope_replacer_0)
    module_0.lazy_import(str_0, str_0)


def test_case_14():
    str_0 = "U\x0cYf?\x0bk=?\x0bp3(U>[[`5r"
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        str_0, str_0, str_0
    )
    module_0.lazy_import(illegal_use_of_scope_replacer_0, str_0)


def test_case_15():
    str_0 = "k;>7J]P~}3S)KV\tL"
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0, str_0: str_0}
    scope_replacer_0 = module_0.ScopeReplacer(dict_0, str_0, str_0)
    module_0.lazy_import(dict_0, scope_replacer_0)
