# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pytutils.lazy.lazy_import as module_0


def test_case_0():
    str_0 = "! q=~T2\x0b;\x0b=a\x0bK2.\x0c+&"
    var_0 = module_0.IllegalUseOfScopeReplacer(str_0, str_0, str_0)
    var_0.__repr__()


def test_case_1():
    str_0 = "C\x0c\x0bj86f69\\qt\nqF]2<"
    var_0 = module_0.IllegalUseOfScopeReplacer(str_0, str_0)


def test_case_2():
    str_0 = "\n    Parses env file content.\n\n    From honcho.\n\n    >>> lines = ['TEST=${HOME}/yeee', 'THISIS=~/a/test', 'YOLO=~/swaggins/$NONEXISTENT_VAR_THAT_DOES_NOT_EXIST']\n    >>> load_env_file(lines, write_environ=dict())\n    OrderedDict([('TEST', '.../yeee'),\n             ('THISIS', '.../a/test'),\n             ('YOLO',\n              '.../swaggins/$NONEXISTENT_VAR_THAT_DOES_NOT_EXIST')])\n\n    "
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(str_0, str_0)
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_3():
    none_type_0 = None
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        none_type_0, none_type_0
    )
    var_0 = illegal_use_of_scope_replacer_0.__eq__(none_type_0)


def test_case_4():
    str_0 = "sub"
    dict_0 = {str_0: str_0}
    import_replacer_0 = module_0.ImportReplacer(dict_0, str_0, dict_0)
    import_replacer_0.__getattribute__(str_0)


def test_case_5():
    str_0 = "BrT@<`,bVf^UgG/fA"
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0}
    import_replacer_0 = module_0.ImportReplacer(dict_0, str_0, dict_0, str_0)
    module_0.lazy_import(import_replacer_0, str_0, str_0)


def test_case_6():
    none_type_0 = None
    module_0.ImportReplacer(none_type_0, none_type_0, none_type_0, none_type_0)


def test_case_7():
    import_processor_0 = module_0.ImportProcessor()


def test_case_8():
    str_0 = ";\x0c\x0bj86f69\\qt\nqF]2<"
    module_0.lazy_import(str_0, str_0, str_0)


def test_case_9():
    str_0 = ".;\x0c\x0bj86fqt\n[qFX]2<"
    module_0.lazy_import(str_0, str_0)


def test_case_10():
    var_0 = module_0.disallow_proxying()


def test_case_11():
    bool_0 = False
    set_0 = {bool_0, bool_0, bool_0, bool_0}
    module_0.ImportReplacer(bool_0, set_0, set_0, bool_0, set_0)


def test_case_12():
    str_0 = "];$1-XV|3]x#BO\rPw"
    module_0.lazy_import(str_0, str_0)


def test_case_13():
    str_0 = "g\"A'(jaYsBC+kTj4| h"
    module_0.lazy_import(str_0, str_0)


def test_case_14():
    str_0 = "BrT@<`,bVf^UgG/fA"
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(str_0, str_0)
    var_0 = illegal_use_of_scope_replacer_0.__eq__(illegal_use_of_scope_replacer_0)
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_15():
    str_0 = "\n    Create a random hex string of a specific length performantly.\n\n    :param int length: length of hex string to generate\n    :return: random hex string\n    "
    module_0.lazy_import(str_0, str_0)


def test_case_16():
    str_0 = ""
    dict_0 = {str_0: str_0}
    import_replacer_0 = module_0.ImportReplacer(dict_0, str_0, dict_0)
    var_0 = module_0.disallow_proxying()
    module_0.lazy_import(str_0, str_0)


def test_case_17():
    str_0 = "\n    Parses env file content.\n\n    From honcho.\n\n    >>> lines = ['TEST=${HOME}/yeee', 'THISIS=~/a/test', 'YOLO=~/swaggins/$NONEXISTENT_VAR_THAT_DOES_NOT_EXIST']\n    >>> load_env_file(lines, write_environ=dict())\n    OrderedDict([('TEST', '.../yeee'),\n             ('THISIS', '.../a/test'),\n             ('YOLO',\n              '.../swaggins/$NONEXISTENT_VAR_THAT_DOES_NOT_EXIST')])\n\n    "
    module_0.lazy_import(str_0, str_0)


def test_case_18():
    str_0 = "\n    Parses env file content.\n\n  i From honcho.\n\n    >>> lines = ['TEST=${HOME}/yeee', 'THISIS=~/a/test', 'YOLO=~/swaggins/$NONEXISTENT_VAR_THAT_DOES_NOT_EXIST']\n    >>> load_env_file(lines, write_environ=dict())\n    OrderedDict([('TEST', '.../yeee'),\n             ('THISIS', '.../a/test'W,\n             ('YOLO',\n              '.../swaggins/$NONEXISTENT_VAR_THAT_DOES_NOT_EXIST')])\n\n    "
    module_0.lazy_import(str_0, str_0)


def test_case_19():
    str_0 = "BrT@<`,bVf^UgG/fA"
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0}
    import_replacer_0 = module_0.ImportReplacer(dict_0, str_0, dict_0, str_0)
    module_0.lazy_import(import_replacer_0, import_replacer_0)
