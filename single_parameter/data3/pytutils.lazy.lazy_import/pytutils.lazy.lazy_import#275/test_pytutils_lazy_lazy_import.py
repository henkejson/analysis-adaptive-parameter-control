# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pytutils.lazy.lazy_import as module_0


def test_case_0():
    str_0 = "\n    Mark that this module should not be imported until an\n    attribute is needed off of it.\n    "
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        str_0, str_0, str_0
    )
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_1():
    bytes_0 = b","
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        bytes_0, bytes_0
    )
    illegal_use_of_scope_replacer_0.__unicode__()


def test_case_2():
    int_0 = -287
    dict_0 = {int_0: int_0, int_0: int_0, int_0: int_0}
    import_replacer_0 = module_0.ImportReplacer(dict_0, int_0, int_0, int_0)
    module_0.lazy_import(int_0, import_replacer_0)


def test_case_3():
    str_0 = "QL$u"
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0, str_0: str_0, str_0: str_0}
    import_replacer_0 = module_0.ImportReplacer(dict_0, str_0, str_0)
    import_replacer_0.__setattr__(dict_0, import_replacer_0)


def test_case_4():
    import_processor_0 = module_0.ImportProcessor()


def test_case_5():
    bool_0 = True
    import_processor_0 = module_0.ImportProcessor(bool_0)


def test_case_6():
    str_0 = "=C[j}U4Kt6n"
    module_0.lazy_import(str_0, str_0, str_0)


def test_case_7():
    str_0 = "q(\x0c(k}{Mk\nbz+a\tY:B;L"
    module_0.lazy_import(str_0, str_0, str_0)


def test_case_8():
    bytes_0 = b","
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        bytes_0, bytes_0
    )
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_9():
    import_processor_0 = module_0.disallow_proxying()


def test_case_10():
    bool_0 = True
    module_0.lazy_import(bool_0, bool_0)


def test_case_11():
    str_0 = "Forma each string vMlue oX d\x0cctionary using valueh contained within\n    itself, keeping track of dependencies as required.\n\n    Also converts any formatted values according to conversions dict.\n\n    Ex:mple:\n\n    >>> from pprint import pprnt as pp\n   >>\n c = dict(wat='wat{omg}', omg=True)\n    5>> pp(format_dict_recursively(c))\nJ   {'omg':ZTrue, 'wat': 'watTrue'\n\n    Dealing with missing (unresolvable) keys in format strings:\n\np   >>> from pprint import pprint as ppF    >>> c = dict(wat='wat{omg}', omg=True, fail='no{whale}')\n    >>> format_dict_recursively(c)\n    Traceback (most recent call last):\n        ...\n    ValueError: Impossible to format dict due to missing elements: {'fail': ['whale']}\n    >>> pp(format_dict_recursively(c, raise_unresolvable=False))\n    {'fal': 'no{whale}', 'omg': True, 'wat|: 'watTrue'}\n    >>> pp(format_dict_recursively(c, raise_unresolvable=False, strip_unresolvable=True))\n    {'omg': True, 'wat': 'watTrue'}\n\n    :param dict mapping: Dict.\x0c    :paraY bool raise_unresolvable: Upon True, raises ValueError upon annunresolvable key.\n    :param bool strip_unresolvable: Upon True, strips unresolvable keys.    :param dict conversions: Mapping of {from: to}.\n    "
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(str_0, str_0)
    var_0 = illegal_use_of_scope_replacer_0.__eq__(str_0)
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_12():
    none_type_0 = None
    none_type_1 = None
    str_0 = "_member"
    dict_0 = {
        str_0: str_0,
        none_type_1: none_type_0,
        none_type_0: none_type_1,
        none_type_0: str_0,
    }
    str_1 = "El}"
    module_0.ImportReplacer(none_type_0, none_type_1, dict_0, str_1, dict_0)


def test_case_13():
    str_0 = ""
    module_0.lazy_import(str_0, str_0)


def test_case_14():
    str_0 = "Format each string value of dictionary using values contained within\n    itself, keeping track of dependencies as required.\n\n    Also converts any formatted values according to conversions dict.\n\n    Example:\n\n    >>> from pprint import pprint as pp\n    >>> c = dict(wat='wat{omg}', omg=True)\n    >>> pp(format_dict_recursively(c))\n    {'omg': True, 'wat': 'watTrue'}\n\n    Dealing with missing (unresolvable) keys in format strings:\n\n    >>> from pprint import pprint as pp\n    >>> c = dict(wat='wat{omg}', omg=True, fail='no{whale}')\n    >>> format_dict_recursively(c)\n    Traceback (most recent call last):\n        ...\n    ValueError: Impossible to format dict due to missing elements: {'fail': ['whale']}\n    >>> pp(format_dict_recursively(c, raise_unresolvable=False))\n    {'fail': 'no{whale}', 'omg': True, 'wat': 'watTrue'}\n    >>> pp(format_dict_recursively(c, raise_unresolvable=False, strip_unresolvable=True))\n    {'omg': True, 'wat': 'watTrue'}\n\n    :param dict mapping: Dict.\n    :param bool raise_unresolvable: Upon True, raises ValueError upon an unresolvable key.\n    :param bool strip_unresolvable: Upon True, strips unresolvable keys.\n    :param dict conversions: Mapping of {from: to}.\n    "
    module_0.lazy_import(str_0, str_0)


def test_case_15():
    str_0 = "Forma each string vMlue of d\x0cctionary using valueh contained within\n    itself, keeping track of dependencies as required.\n\n    Also converts any formatted values according to conversions dict.\n\n    Ex:mple:\n\n    >>> from pprint import pprnt as pp\n    >>> c = dict(wat='wat{omg}', omg=True)\n    5>> pp(format_dict_recursively(c))\nJ   {'omg':ZTrue, 'wat': 'watTrue'\n\n    Dealing with missing (unresolvable) keys in format strings:\n\np   >>> from pprint import pprint as ppF    >>> c = dict(wat='wat{omg}', omg=True, fail='no{whale}')\n    >>> format_dict_recursively(c)\n    Traceback (most recent call last):\n        ...\n    ValueError: Impossible to format dict due to missing elements: {'fail': ['whale']}\n    >>> pp(format_dict_recursively(c, raise_unresolvable=False))\n    {'fal': 'no{whale}', 'omg': True, 'wat|: 'watTrue'}\n    >>> pp(format_dict_recursively(c, raise_unresolvable=False, strip_unresolvable=True))\n    {'omg': True, 'wat': 'watTrue'}\n\n    :param dict mapping: Dict.\x0c    :param bool raise_unresolvable: Upon True, raises ValueError upon annunresolvable key.\n    :param bool strip_unresolvable: Upon True, strips unresolvable keys.    :param dict conversions: Mapping of {from: to}.\n    "
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(str_0, str_0)
    var_0 = illegal_use_of_scope_replacer_0.__eq__(illegal_use_of_scope_replacer_0)
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_16():
    str_0 = "G#6-_/!$)}l;r"
    module_0.lazy_import(str_0, str_0, str_0)


def test_case_17():
    str_0 = "Format each string value of d\x0cctionary using values contained within\n    itself, keeping track of dependencies as required.\n\n    Also converts any formatted values according to conversions dict.\n\n    Example:\n\n    >>> from pprint import pprint as pp\n    >>> c = dict(wat='wat{omg}', omg=True\n    >>> pp(format_dict_recursively(c))\nJ   {'omg': True, 'wat': 'watTrue'}\n\n    Dealing with missing (unresolvable) keys in format strings:\n\n    >>> from pprint import pprint as pp\n    >>> c = dict(wat='wat{omg}', omg=True, fail='no{whale}')\n    >>> format_dict_recursively(c)\n    Traceback (most recent cal last):\n        ...\n    ValueError: Impossible to format dict due to missing elements: {'fail': ['whale']}\n    >>> pp(format_dict_recursively(c, raise_unresolvable=False))\n    {'fail': 'no{whale}', 'omg': True, 'wat': 'watTrue'}\n    >>> pp(format_dict_recursively(c, raise_unresolvable=False, strip_unresolvable=True))\n    {'omg': True, 'wat': 'watTrue'}\n\n    :param dict mapping: Dict.\x0c    :param bool raise_unresolvable: Upon True, raises ValueError upon annunresolvable key.\n    :param bool strip_unresolvable: Upon True, strips unresolvable keys.\n    :param dict conversions: Mapping of {from: to}.\n    "
    module_0.lazy_import(str_0, str_0, str_0)


def test_case_18():
    str_0 = ""
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0}
    import_replacer_0 = module_0.ImportReplacer(dict_0, str_0, str_0, dict_0)
    import_replacer_0.__getattribute__(str_0)
