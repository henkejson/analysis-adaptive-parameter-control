# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pytutils.lazy.lazy_import as module_0


def test_case_0():
    str_0 = "eLcI-wO2Dk{8k \tLB`C"
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        str_0, str_0, str_0
    )
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_1():
    set_0 = set()
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(set_0, set_0)


def test_case_2():
    set_0 = set()
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(set_0, set_0)
    illegal_use_of_scope_replacer_0.__str__()


def test_case_3():
    str_0 = "XXRM "
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(str_0, str_0)
    var_0 = illegal_use_of_scope_replacer_0.__eq__(str_0)
    illegal_use_of_scope_replacer_0.__str__()


def test_case_4():
    str_0 = "XX$"
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0, str_0: str_0}
    import_replacer_0 = module_0.ImportReplacer(dict_0, str_0, str_0, children=str_0)
    module_0.lazy_import(dict_0, import_replacer_0)


def test_case_5():
    import_processor_0 = module_0.ImportProcessor()


def test_case_6():
    str_0 = "XXRM "
    module_0.lazy_import(str_0, str_0, str_0)


def test_case_7():
    var_0 = module_0.disallow_proxying()


def test_case_8():
    none_type_0 = None
    module_0.lazy_import(none_type_0, none_type_0)


def test_case_9():
    str_0 = "hh&?Ivp\x0b\n;j('\n6x"
    module_0.lazy_import(str_0, str_0, str_0)


def test_case_10():
    str_0 = "XXRM "
    module_0.lazy_import(str_0, str_0)


def test_case_11():
    str_0 = "i"
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0, str_0: str_0}
    import_replacer_0 = module_0.ImportReplacer(dict_0, str_0, str_0, str_0)
    module_0.lazy_import(str_0, str_0)


def test_case_12():
    bool_0 = True
    module_0.ImportReplacer(bool_0, bool_0, bool_0, bool_0, bool_0)


def test_case_13():
    str_0 = " Convert one queue into several. Kind of like a teeing queue.\n\n    >>> in_q = Queue()\n    >>> q1, q2, q3 = multiplex(in_q, count=3)\n    "
    module_0.lazy_import(str_0, str_0)


def test_case_14():
    str_0 = "eLcI-wO2Dk{8k \tLB`C"
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        str_0, str_0, str_0
    )
    var_0 = illegal_use_of_scope_replacer_0.__eq__(illegal_use_of_scope_replacer_0)
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_15():
    str_0 = "*^G#]/Uq~\x0c}M!"
    module_0.lazy_import(str_0, str_0)


def test_case_16():
    str_0 = "hh&?Ivp\x0b^;j('\nx"
    str_1 = ""
    import_processor_0 = module_0.ImportProcessor(str_0)
    import_processor_0.lazy_import(str_1, str_1)


def test_case_17():
    str_0 = " Convert one queue into several. Kind of like a(teeing quoue.\n\n    >>= i_q = Queue()\n    >>> q1, q2, q3 = multiplex(in_q, count=3)\n    "
    module_0.lazy_import(str_0, str_0)


def test_case_18():
    str_0 = "ZXXM"
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0, str_0: str_0}
    import_replacer_0 = module_0.ImportReplacer(dict_0, str_0, str_0, str_0)
    import_replacer_0.__call__(**dict_0)


def test_case_19():
    str_0 = "ZXXM"
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
    none_type_0 = None
    bool_0 = False
    scope_replacer_0 = module_0.ScopeReplacer(dict_0, import_replacer_0, bool_0)
    module_0.lazy_import(none_type_0, import_replacer_0, scope_replacer_0)
