# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pytutils.lazy.lazy_import as module_0
import builtins as module_1


def test_case_0():
    str_0 = (
        "Passing 'typed' to cachedmethod() is deprecated, use 'key=typedkey' instead"
    )
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        str_0, str_0, str_0
    )
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_1():
    var_0 = module_0.disallow_proxying()
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(var_0, var_0)
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_2():
    str_0 = "j9t<tL"
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0}
    import_replacer_0 = module_0.ImportReplacer(dict_0, str_0, str_0, str_0)
    import_replacer_0.__getattribute__(import_replacer_0)


def test_case_3():
    var_0 = module_0.disallow_proxying()
    none_type_0 = None
    module_0.ImportReplacer(var_0, none_type_0, none_type_0)


def test_case_4():
    import_processor_0 = module_0.ImportProcessor()


def test_case_5():
    str_0 = (
        "Passing 'typed' to cachedmethod() is deprecated, use 'key=typedkey' instead"
    )
    module_0.lazy_import(str_0, str_0, str_0)


def test_case_6():
    str_0 = (
        "Passing 'typed' to cachedmethod() is deprecated, use 'key=typedkey' instead"
    )
    module_0.lazy_import(str_0, str_0)


def test_case_7():
    var_0 = module_0.disallow_proxying()


def test_case_8():
    str_0 = "1H) o"
    module_0.lazy_import(str_0, str_0)


def test_case_9():
    str_0 = (
        "Passing 'typed' to cachedmethod() is deprecated, use 'key*typedkey' instead"
    )
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(str_0, str_0)
    var_0 = illegal_use_of_scope_replacer_0.__eq__(illegal_use_of_scope_replacer_0)
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_10():
    none_type_0 = None
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        none_type_0, none_type_0
    )
    var_0 = illegal_use_of_scope_replacer_0.__eq__(none_type_0)
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_11():
    var_0 = module_0.disallow_proxying()
    import_processor_0 = module_0.ImportProcessor(var_0)
    bytes_0 = b"\x9e\xb1\xf7\x9b\xfe\x91\xd9\x9b+"
    module_0.ImportReplacer(var_0, var_0, var_0, bytes_0, bytes_0)


def test_case_12():
    bool_0 = True
    var_0 = module_1.Exception()
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        bool_0, var_0, bool_0
    )
    var_1 = var_0.__str__()
    module_0.lazy_import(var_1, var_1)


def test_case_13():
    str_0 = (
        "\n        A standin for a module to prevent it from being imported\n        "
    )
    module_0.lazy_import(str_0, str_0)


def test_case_14():
    str_0 = "jmTsm9K#yIbw\r\x0be"
    module_0.lazy_import(str_0, str_0)


def test_case_15():
    str_0 = "Passing 'typed' to (achedmethod( is deprecated, use 'key=typedkey' instead"
    module_0.lazy_import(str_0, str_0, str_0)


def test_case_16():
    str_0 = (
        "Passing 'typed' to cachedmethod(c is deprecated, use 'key=typedkey' inste\nd"
    )
    var_0 = str_0.__str__()
    module_0.lazy_import(var_0, str_0)


def test_case_17():
    str_0 = "\n    Loads (and returns) an env file specified by `filename` into the mapping `environ`.\n\n    >>> lines = ['TEST=${HOME}/yeee-$PATH', 'THISIS=~/a/test', 'YOLO=~/swaggins/$NONEXISTENT_VAR_THAT_DOES_NOT_EXIST']\n    >>> load_env_file(lines, write_environ=dict())\n    OrderedDict([('TEST', '.../.../yeee-...:...'),\n             ('THISIS', '.../a/test'),\n             ('YOLO',\n              '.../swaggins/$NONEXISTENT_VAR_THAT_DOES_NOT_EXIST')])\n    "
    dict_0 = {str_0: str_0, str_0: str_0}
    module_0.lazy_import(dict_0, str_0)


def test_case_18():
    dict_0 = {}
    none_type_0 = None
    import_replacer_0 = module_0.ImportReplacer(
        dict_0, none_type_0, dict_0, none_type_0
    )
    module_0.lazy_import(none_type_0, import_replacer_0)


def test_case_19():
    str_0 = "j9t<tL"
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0}
    import_replacer_0 = module_0.ImportReplacer(dict_0, str_0, dict_0)
    import_replacer_0.__getattribute__(import_replacer_0)


def test_case_20():
    str_0 = "from C"
    module_0.lazy_import(str_0, str_0, str_0)
