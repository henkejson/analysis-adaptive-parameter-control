# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pytutils.lazy.lazy_import as module_0


def test_case_0():
    str_0 = "Make lazy_compile the default compile mode for regex compilation.]\n    This overrides re.compile with lazy_compile. To restore the original\n    functionality, call reset_compile().\n    "
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        str_0, str_0, str_0
    )
    illegal_use_of_scope_replacer_0.__str__()


def test_case_1():
    none_type_0 = None
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        none_type_0, none_type_0
    )


def test_case_2():
    none_type_0 = None
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        none_type_0, none_type_0
    )
    illegal_use_of_scope_replacer_0.__str__()


def test_case_3():
    str_0 = "Make lazy_compile the default compile mode for regex compilation.\n\n    This overrides re.compile with lazy_compile. To restore the original\n    functionality, call reset_compile().\n    "
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(str_0, str_0)
    var_0 = illegal_use_of_scope_replacer_0.__eq__(str_0)
    module_0.ImportReplacer(
        illegal_use_of_scope_replacer_0, illegal_use_of_scope_replacer_0, str_0
    )


def test_case_4():
    str_0 = "\n\x0bu"
    var_0 = module_0.disallow_proxying()
    import_processor_0 = module_0.ImportProcessor()
    var_1 = module_0.disallow_proxying()
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(var_1, var_1)
    var_2 = illegal_use_of_scope_replacer_0.__eq__(var_0)
    module_0.ImportReplacer(var_1, str_0, var_1, illegal_use_of_scope_replacer_0, var_1)


def test_case_5():
    dict_0 = {}
    list_0 = []
    module_0.ImportReplacer(dict_0, dict_0, list_0, children=dict_0)


def test_case_6():
    import_processor_0 = module_0.ImportProcessor()


def test_case_7():
    str_0 = "&fJ'<;m="
    module_0.lazy_import(str_0, str_0)


def test_case_8():
    var_0 = module_0.disallow_proxying()


def test_case_9():
    str_0 = "Make lazy_compile the default compile mode for regex compilation.]\n    This overrides re.compile with lazy_compile. To restore the original\n    functionality, call reset_compile().\n    "
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        str_0, str_0, str_0
    )
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_10():
    str_0 = "zJ'<;m="
    module_0.lazy_import(str_0, str_0, str_0)


def test_case_11():
    str_0 = "Make lazy_compile the default compile mode for regex compilation.\n\n    This overrides re.compile with lazy_compile. To restore the original\n    functionality, call reset_compile().\n    "
    module_0.lazy_import(str_0, str_0)


def test_case_12():
    str_0 = "Make lazy_c(mpile the default compile mode for regex compilation.]\n    This overrides re.compile w'th lazy_compile. To restore the original\n    functonality, call reset_]ompile().\n    "
    module_0.lazy_import(str_0, str_0)


def test_case_13():
    str_0 = "E}LpHk#>,WpCIL\nz\r"
    module_0.lazy_import(str_0, str_0, str_0)


def test_case_14():
    str_0 = "\n    Loads (and returns) an env file specified by `filename` into the mapping `environ`.\n\n    >>> lines = ['TEST=${HOME}/yeee-$PATH', 'THISIS=~/a/test', 'YOLO=~/swaggins/$NONEXISTENT_VAR_THAT_DOES_NOT_EXIS']\n    >>> load_env_file(lines, write_environ=dict())\n    OrderedDict([('TEST', '.../.../yeee-...:...'),\n             ('THISIS', '.../a/test'),\n     l       ('YOLO',\n              '.../swaggins/$NONEXISTENT_VAR_THAT_DES_NOT_EXIST')])\n    "
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(str_0, str_0)
    var_0 = illegal_use_of_scope_replacer_0.__eq__(illegal_use_of_scope_replacer_0)
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_15():
    str_0 = "\n"
    module_0.lazy_import(str_0, str_0)


def test_case_16():
    str_0 = ")"
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0}
    module_0.ImportReplacer(dict_0, dict_0, str_0, str_0, str_0)


def test_case_17():
    dict_0 = {}
    none_type_0 = None
    scope_replacer_0 = module_0.ScopeReplacer(dict_0, dict_0, none_type_0)
    scope_replacer_0.lazy_import(dict_0, dict_0)


def test_case_18():
    str_0 = "\n    Loads (and returns) an env file specified by `filename` into the mapping `environ`.\n\n    >>> lines = ['TEST=${HOME}/yeee-$PATH', 'THISIS=~/a/test', 'YOLO=~/swaggins/$NONEXISTENT_VAR_THAT_DOES_NOT_EXIST']\n    >>> load_env_file(lines, write_environ=dict())\n    OrderedDict([('TEST', '.../.../yeee-...:...'),\n             ('THISIS', '.../a/test'),\n             ('YOLO',\n              '.../swaggins/$NONEXISTENT_VAR_THAT_DOES_NOT_EXIST')])\n    "
    module_0.lazy_import(str_0, str_0)
