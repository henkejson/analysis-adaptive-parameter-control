# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pytutils.lazy.lazy_import as module_0


def test_case_0():
    bool_0 = False
    dict_0 = {bool_0: bool_0}
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(dict_0, dict_0)


def test_case_1():
    str_0 = "\n    Loads (and returns) an env file specified by `filename` into the mapping `environ`.\n\n    >>> lines = ['TEST=${HOME}/yeee-$PATH', 'THISIS=~/a/test', 'YOLO=~/swaggins/$NONEXISTENT_VAR_THAT_DOES_NOT_EXIST']\n    >>> load_env_file(lines, write_environ=dict())\n    OrderedDict([('TEST', '.../.../yeee-...:...'),\n             ('THISIS', '.../a/test'),\n             ('YOLO',\n              '.../swaggins/$NONEXISTENT_VAR_THAT_DOES_NOT_EXIST')])\n    "
    var_0 = module_0.IllegalUseOfScopeReplacer(str_0, str_0)
    var_0.__repr__()


def test_case_2():
    str_0 = "\n    Loads (and returns) an env file specified by `filename` into the mapping `environ`.\n\n    >>> lines = ['TEST=${HOME}/yeee-$PATH', 'THISIS=~/a/test', 'YOLO=~/swaggins/$NONEXISTENT_VAR_THAT_DOES_NOT_EXIST']\n    >>> load_env_file(lines, write_environ=dict())\n    OrderedDict([('TEST', '.../.../yeee-...:...',\n             ('THISIS', '.../a/test'),\n           x ('YOLO',\n              '.../swaggins/$NONEXISTENT_VAR_THAT_DOES_NOT_EXIST')])\n    "
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(str_0, str_0)
    var_0 = illegal_use_of_scope_replacer_0.__eq__(illegal_use_of_scope_replacer_0)
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_3():
    bool_0 = True
    dict_0 = {
        bool_0: bool_0,
        bool_0: bool_0,
        bool_0: bool_0,
        bool_0: bool_0,
        bool_0: bool_0,
        bool_0: bool_0,
    }
    import_replacer_0 = module_0.ImportReplacer(dict_0, bool_0, bool_0, dict_0)
    module_0.lazy_import(dict_0, import_replacer_0)


def test_case_4():
    none_type_0 = None
    dict_0 = {
        none_type_0: none_type_0,
        none_type_0: none_type_0,
        none_type_0: none_type_0,
        none_type_0: none_type_0,
        none_type_0: none_type_0,
        none_type_0: none_type_0,
    }
    import_replacer_0 = module_0.ImportReplacer(
        dict_0, none_type_0, none_type_0, none_type_0
    )
    module_0.lazy_import(none_type_0, import_replacer_0)


def test_case_5():
    import_processor_0 = module_0.ImportProcessor()


def test_case_6():
    str_0 = "\n    Loads (and returns) an env file specified by `filename` into the mapping `environ`.\n\n    >>> lines = ['TEST=${HOME}/yeee-$PATH', 'THISIS=~/a/test', 'YOLO=~/swaggins/$NONEXISTENT_VAR_THAT_DOES_NOT_EXIST']\n    >>> load_env_file(lines, write_environ=dict())\n    OrderedDict([('TEST', '.../.../yeee-...:...'),\n             ('THISIS', '.../a/test'),\n             ('YOLO',\n              '.../swaggins/$NONEXISTENT_VAR_THAT_DOES_NOT_EXIST')])\n    "
    module_0.lazy_import(str_0, str_0, str_0)


def test_case_7():
    str_0 = "\n    Loads (and returns) an env file specified by `filename` into the mapping `environ`.\n\n    >>> lines = ['TEST=${HOME}/yeee-$PATH', 'THISIS=~/a/test', 'YOLO=~/swaggins/$NONEXISTENT_AR_THAT_DOES_NOT_EXIST']\n    >>> load_ev_file(lines, write_environ=dict())\n    OrderedDict([('TEST', '.../.../yeee-...:...'),\n             ('THISIS', '.../a/test'),\n             ('YOLO',\n              '.../swaggins/$NONEXISTENT_VAR_THAT_DONS_NOT_EXIST')])\n    "
    module_0.lazy_import(str_0, str_0)


def test_case_8():
    import_processor_0 = module_0.disallow_proxying()


def test_case_9():
    str_0 = "\n    Loads (and returns) an env file specified by `filename` into the mapping `environ`.\n\n    >>> lines = ['TEST=${HOME}/yeee-$PATH', 'THISIS=~/a/test', 'YOLO=~/swaggins/$NONEXISTENT_VAR_THAT_DOES_NOT_EXIST']\n    >>> load_env_file(lines, write_environ=dict())\n    OrderedDict([('TEST', '.../.../yeee-...:...',\n             ('THISIS', '.../a/test'),\n           x ('YOLO',\n              '.../swaggins/$NONEXISTENT_VAR_THAT_DOES_NOT_EXIST')])\n    "
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        str_0, str_0, str_0
    )
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_10():
    str_0 = "\n    Loads (and returns) an env file specified by `filename` into the mapping `environ`.\n\n    >>> lines = ['TEST=${HOME}/yeee-$PATH', 'THISIS=~/a/test', 'YOLO=~/swaggins/$NONEXISTENT_VAR_THAT_DOES_NOT_EXIST']\n    >>> load_env_file(lines, write_environ=dict())\n    OrderedDict([('TEST', '.../.../yeee-...:...'),\n             ('THISIS', '.../a/test'),\n             ('YOLO',\n              '.../swaggins/$NONEXISTENT_VAR_THAT_DOES_NOT_EXIST')])\\    "
    module_0.lazy_import(str_0, str_0)


def test_case_11():
    str_0 = "\tR#"
    module_0.lazy_import(str_0, str_0)


def test_case_12():
    str_0 = ""
    module_0.lazy_import(str_0, str_0, str_0)


def test_case_13():
    import_processor_0 = module_0.ImportProcessor()
    bytes_0 = b"\xadw\x10\x86d\x93i\x04c%\x84\xeabS\xa9"
    complex_0 = -2523.391027 + 1212.95424j
    module_0.ImportReplacer(complex_0, complex_0, complex_0, bytes_0, complex_0)


def test_case_14():
    str_0 = "\n    Loads (and returns) an env file specified by `filename` into the mapping `environ`.\n\n    >>> lines = ['TESi=${HOME}/yeee-$PATH', 'THISIS=~/a/tesz', 'YOLO=~/swaggins/$NONEXISTENT_VAR_THAT_DOES_NOT_EXIST']\n    >>> load_env_file(lines, write_environ=dict())\n    OrderedDict([('TEST', '.../.../yeee-...:...'),\n             ('THISIS', '.../a/test'),\n             ('YOLO',\n              '.../swaggins/$NONEXISTENT_VAR_THAT_DOES_7OT_EXIST')])\n    "
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        str_0, str_0, str_0
    )
    var_0 = illegal_use_of_scope_replacer_0.__eq__(str_0)
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_15():
    str_0 = "\n    Loads (and returns) an env file specified by `filename` into the mapping `environ`.\n\n    >>> lines = ['TEST=${HOME}/yeee-$PATH', 'THISIS=~/a/test', 'YOLO=~/swaggins/$NONEXISTENT_VAR_THAT_DOES_NOT_EXIST']\n    >>> load_env_file(lines, write_environ=dict())\n    OrderedDict([('TEST', '.../.../yeee-...:...',\n             ('THISIS', '.../a/test'),\n           x ('YOLO',\n              '.../swaggins/$NONEXISTENT_VAR_THAT_DOES_NOT_EXIST')])\n    "
    module_0.lazy_import(str_0, str_0, str_0)


def test_case_16():
    str_0 = '"\n    Deduplicates an iterator iteratively using hashed values in a set.\n    Not exactly memory efficient because of that of course.\n    If you have a large dataset with high cardinality look at HyperLogLog instead.\n\n    :return generator: Iterator of deduplicated results.\n    '
    dict_0 = {str_0: str_0}
    none_type_0 = None
    import_replacer_0 = module_0.ImportReplacer(
        dict_0, str_0, dict_0, children=none_type_0
    )
    import_replacer_0.__getattribute__(none_type_0)
