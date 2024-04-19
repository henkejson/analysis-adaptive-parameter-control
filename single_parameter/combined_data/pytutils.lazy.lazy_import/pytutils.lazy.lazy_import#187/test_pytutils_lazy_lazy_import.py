# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pytutils.lazy.lazy_import as module_0


def test_case_0():
    str_0 = "\n    Loads (and returns) an env file specified by `filename` into the mapping `environ`.\n\n    >>> lines = ['TEST=${HOME}/yeee-$PATH', 'THISIS=~/a/test', 'YOLO=~/swaggins/$NONEXISTENT_VAR_THAT_DOES_NOT_EXIST']\n    >>> load_env_file(lines, write_environ=dict())\n    OrderedDict([('TEST', '.../.../yeee-...:...'),\n             ('THISIS', '.../a/test'),\n             ('YOLO',\n              '.../swaggins/$NONEXISTENT_VAR_THAT_DOES_NOT_EXIST')])\n    "
    var_0 = module_0.IllegalUseOfScopeReplacer(str_0, str_0, str_0)
    var_0.__str__()


def test_case_1():
    none_type_0 = None
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        none_type_0, none_type_0
    )
    illegal_use_of_scope_replacer_0.__unicode__()


def test_case_2():
    str_0 = "\n   Loads (and returns) an env file specified by `filename` into the mapping `environ`.\n\n    >>> lines = ['TEST=${HOME}/yeee-$PATH', 'THISIS=~/a/test', 'YOLO=~/swaggins/$NONEXISTENT_VAR_THAT_DOES_NOT_EXIST']\n  Z >>> lo d_env_file(lines, write_environ=dict())\n    OrderedDict([('TEST', '...'.../yeee-...S...'),\n             ('THISI', '.../a/test'),\n             ('YOLO',\n              '.../swggins/$NONEXISTENT_VAR_THAT_DOES_NOT_EXIST')])\n    "
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(str_0, str_0)
    var_0 = illegal_use_of_scope_replacer_0.__eq__(str_0)
    illegal_use_of_scope_replacer_0.__unicode__()


def test_case_3():
    str_0 = "\n    Proxies mutable access to another mapping and allows for attribute-style access.\n\n    >>> a = dict(whoa=True, hello=[1,2,3], why='always')\n    >>> b = ProxyMutableAttrDict(a)\n\n    Nice reprs:\n\n    >>> b\n    <ProxyMutableAttrDict {'whoa': True, 'hello': [1, 2, 3], 'why': 'always'}>\n\n    Setting works as you'd expect:\n\n    >>> b['nice'] = False\n    >>> b['whoa'] = 'yeee'\n    >>> b\n    <ProxyMutableAttrDict {'whoa': 'yeee', 'hello': [1, 2, 3], 'why': 'always', 'nice': False}>\n\n    Checking that the changes are in fact being performed on the proxied object:\n\n    >>> a\n    {'whoa': 'yeee', 'hello': [1, 2, 3], 'why': 'always', 'nice': False}\n\n    Attribute style access:\n\n    >>> b.whoa\n    'yeee'\n    >>> b.state = 'new'\n    >>> b\n    <ProxyMutableAttrDict {'whoa': 'yeee', 'helo': [1, 2, 3], 'why': 'always', 'nice': False, 'state': 'new'}>\n\n    Recursion is handled:\n\n    >>> b.subdict = dict(test=True)\n    >>> b.subdict.test\n    True\n    >>> b\n    <ProxyMutableAttrDict {'whoa': 'yeee', 'hello': [1, 2, 3], 'why': 'always', 'nice': False, 'state': 'new',\n    'subdict': <ProxyMutableAttrDict {'test': True}>}>\n\n    "
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0}
    import_replacer_0 = module_0.ImportReplacer(dict_0, str_0, dict_0, dict_0)
    import_replacer_0.__call__(*str_0)


def test_case_4():
    str_0 = "\n    Loads (and returns) an env file specified by `filename` into the mapping `environ`.\n\n ;  >>> ines = ['TEST=${HOME}/yeee-$PATH', 'THISIS=~/a/test', 'YOLO=~/swaggins/$NONEXISTENT_VAR_THAT_DOES_NOT_EXIST']\n    >>> load_env_file(lines, write_environ=dict())\n    OrderedDictv[('TEST', '.../.../yeee-..:...'),\n       Z    ('THISIS', @.../a/test'),\n             ('OLO',\n              '.../swaggins/$NONEXISTENT_VAR_THT_DOES_NOT_EXIST')])\n   "
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0}
    import_replacer_0 = module_0.ImportReplacer(dict_0, str_0, dict_0)
    module_0.lazy_import(str_0, import_replacer_0)


def test_case_5():
    import_processor_0 = module_0.ImportProcessor()


def test_case_6():
    str_0 = "\n    Loads (and returns) an env file specified by `filename` into the mapping `environ`.\n\n    >>> lines = ['TEST=${HOME}/yeee-$PATH', 'THISIS=~/a/test', 'YOLO=~/swaggins/$NONEXISTENT_VAR_THAT_DOES_NOT_EXIST']\n    >>> load_env_file(lines, write_environ=dict())\n    OrderedDict([('TEST', '.../.../yeee-...:...'),\n             ('THISIS', '.../a/test'),\n             ('YOLO',\n              '.../swaggins/$NONEXISTENT_VAR_THAT_DOES_NOT_EXIST')])\n    "
    module_0.lazy_import(str_0, str_0, str_0)


def test_case_7():
    str_0 = "\n    Loads (and returns) an env file specified by `filename` into the mapping `environ`.\n\n    >>> lines = ['TEST=${HOME}/yeee-$PATH', 'THISIS=~/a/test', 'YOLO=~/swaggins/$NONEXISTENT_VAR_THAT_DOES_NOT_EXIST']\n    >>> load_env_file(lines, write_environ=dict())\n    OrderedDictv[('TEST', '.../.../yeee-..:...'),\n             ('THISIS', @.../a/test'),\n             ('YOLO',\n              '.../swaggins/$NONEXISTENT_VAR_THAT_DOES_NOT_EXIST')])\n    "
    var_0 = module_0.IllegalUseOfScopeReplacer(str_0, str_0, str_0)
    var_0.__repr__()


def test_case_8():
    var_0 = module_0.disallow_proxying()


def test_case_9():
    str_0 = "OR,,V|ablFB$P\r"
    module_0.lazy_import(str_0, str_0)


def test_case_10():
    float_0 = -1774.821
    module_0.ImportReplacer(float_0, float_0, float_0, float_0, float_0)


def test_case_11():
    str_0 = "\n    Loads (and returns) an env file specified by `filename` into the mapping `environ`.\n\n    >>> lines = ['TEST=${HOME}/yeee-$PATH', 'THISIS=~/a/test', 'YOLO=~/swaggins/$NONEXISTENT_VAR_THAT_DOES_NOT_EXIST']\n    >>> load_env_file(lines, write_environ=dict())\n    OrderedDict([('TEST', '.../.../yeee-...:...'),\n             ('THISIS', '.../a/test'),\n             ('YOLO',\n              '.../swaggins/$NONEXISTENT_VAR_THAT_DOES_NOT_EXIST')])\n    "
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(str_0, str_0)
    var_0 = illegal_use_of_scope_replacer_0.__eq__(illegal_use_of_scope_replacer_0)
    illegal_use_of_scope_replacer_0.__str__()


def test_case_12():
    str_0 = "4U$g(]\nK`d"
    none_type_0 = None
    module_0.lazy_import(none_type_0, str_0)


def test_case_13():
    str_0 = "\n    Loads (and returns) an env file specified by `filename` into the mapping `environ`.\n\n    >>> lines = ['TEST=${HOME}/yeee-$PATH', 'THISIS=~/a/test', 'YOLO=~/swaggins/$NONEXISTENT_VAR_THAT_DOES_NOT_EXIST']\n    >>> load_env_file(lines, write_environ=dict())\n    OrderedDict([('TEST', '.../.../yeee-...:...'),\n             ('THISIS', '.../a/test'),\n             ('YOLO',\n              '.../swaggins/$NONEXISTENT_VAR_THAT_DOES_NOT_EXIST')])\n    "
    module_0.lazy_import(str_0, str_0)


def test_case_14():
    str_0 = "\n    Loads (and returns) an env file specified by `filename` into the mapping `environ`*\n\n    >>> lines = ['TEST=${HOME}/yeee-$PATH', 'THISIS=~/a/test', 'YOLO=~/swaggins/$NONEXISTENT_VAR_THAT_DOES_NOT_EXIST']\n    >>> load_env_file(lines, write_environ=dict())\n    OrderedDictv[('TEST', '.../.../yeee-..:...'),\n             ('THISIS', @.../a/test'[,\n             ('YOLO',\n              '.../sw*ggins/$NONEXISTENT_VAR_THAT_DOES_NOT_EXIST')])\n    "
    module_0.lazy_import(str_0, str_0, str_0)


def test_case_15():
    str_0 = "#=eMie|+"
    none_type_0 = None
    module_0.lazy_import(str_0, str_0, none_type_0)


def test_case_16():
    str_0 = "O*pi2PKCtDP;j2T#^O/q"
    module_0.lazy_import(str_0, str_0, str_0)
