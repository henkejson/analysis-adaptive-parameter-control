# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pytutils.lazy.lazy_import as module_0


def test_case_0():
    str_0 = "TiX(mR,'|Z&a(L"
    var_0 = module_0.IllegalUseOfScopeReplacer(str_0, str_0, str_0)
    var_0.__repr__()


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
    illegal_use_of_scope_replacer_0.__unicode__()


def test_case_3():
    none_type_0 = None
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        none_type_0, none_type_0
    )
    var_0 = illegal_use_of_scope_replacer_0.__eq__(none_type_0)
    illegal_use_of_scope_replacer_0.__unicode__()


def test_case_4():
    var_0 = module_0.disallow_proxying()
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(var_0, var_0)
    var_1 = illegal_use_of_scope_replacer_0.__eq__(illegal_use_of_scope_replacer_0)
    illegal_use_of_scope_replacer_0.__unicode__()


def test_case_5():
    bool_0 = False
    var_0 = module_0.disallow_proxying()
    dict_0 = {var_0: var_0, bool_0: bool_0}
    import_replacer_0 = module_0.ImportReplacer(dict_0, var_0, var_0, bool_0)
    import_replacer_0.__getattribute__(import_replacer_0)


def test_case_6():
    bool_0 = False
    list_0 = [bool_0, bool_0]
    import_replacer_0 = module_0.ImportReplacer(list_0, bool_0, bool_0, children=bool_0)
    import_replacer_0.__getattribute__(list_0)


def test_case_7():
    import_processor_0 = module_0.ImportProcessor()


def test_case_8():
    str_0 = "[9V"
    module_0.lazy_import(str_0, str_0, str_0)


def test_case_9():
    str_0 = "u9V"
    module_0.lazy_import(str_0, str_0)


def test_case_10():
    var_0 = module_0.disallow_proxying()


def test_case_11():
    bytes_0 = b"\xf2\x14\xa7J+\xc5\xd6\xc2\xc7O\xf9X\xda"
    module_0.ImportReplacer(bytes_0, bytes_0, bytes_0, bytes_0, bytes_0)


def test_case_12():
    str_0 = "\n    Pretty prints with coloring.\n\n    Works in iPython, but not bpython as it does not write directly to term\n    and decodes it instead.\n    "
    module_0.lazy_import(str_0, str_0)


def test_case_13():
    str_0 = "#?ZROEB7"
    import_processor_0 = module_0.disallow_proxying()
    module_0.lazy_import(import_processor_0, str_0, import_processor_0)


def test_case_14():
    str_0 = "&o\n]:y]+AI()^OsD5 "
    module_0.lazy_import(str_0, str_0, str_0)


def test_case_15():
    str_0 = "$gA\\(uY:K8F^[5t?HH"
    module_0.lazy_import(str_0, str_0)


def test_case_16():
    str_0 = '}*;Q/#3O>nj"^ouVp'
    module_0.lazy_import(str_0, str_0)


def test_case_17():
    str_0 = "Upon request import 'module_path' as the name 'module_name'.\n        When imported, prepare children to also be imported.\n\n        :param scope: The scope that objects should be imported into.\n            Typically this is globals()\n        :param name: The variable name. Often this is the same as the\n            module_path. 'bzrlib'\n        :param module_path: A list for the fully specified module path\n            ['bzrlib', 'foo', 'bar']\n        :param member: The member inside the module to import, often this is\n            None, indicating the module is being imported.\n        :param children: Children entries to be imported later.\n            This should be a map of children specifications.\n            ::\n            \n                {'foo':(['bzrlib', 'foo'], None,\n                    {'bar':(['bzrlib', 'foo', 'bar'], None {})})\n                }\n\n        Examples::\n\n            import foo => name='foo' module_path='foo',\n                          member=None, children={}\n            import foo.bar => name='foo' module_path='foo', member=None,\n                              children={'bar':(['foo', 'bar'], None, {}}\n            from foo import bar => name='bar' module_path='foo', member='bar'\n                                   children={}\n            from foo import bar, baz would get translated into 2 import\n            requests. On for 'name=bar' and one for 'name=baz'\n        "
    module_0.lazy_import(str_0, str_0, str_0)
