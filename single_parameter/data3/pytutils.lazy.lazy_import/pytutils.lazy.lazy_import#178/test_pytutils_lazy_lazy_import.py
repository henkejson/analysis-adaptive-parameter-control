# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pytutils.lazy.lazy_import as module_0


def test_case_0():
    str_0 = "\rdI<H3Pmg~p"
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        str_0, str_0, str_0
    )


def test_case_1():
    int_0 = -782
    none_type_0 = None
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        int_0, none_type_0, none_type_0
    )


def test_case_2():
    str_0 = "\rdI<H3Pmg~p"
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        str_0, str_0, str_0
    )
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_3():
    str_0 = 'Pu~_;\n!]B"M7'
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0}
    import_replacer_0 = module_0.ImportReplacer(dict_0, str_0, dict_0, str_0)
    module_0.lazy_import(dict_0, str_0, dict_0)


def test_case_4():
    none_type_0 = None
    module_0.ImportReplacer(none_type_0, none_type_0, none_type_0)


def test_case_5():
    import_processor_0 = module_0.ImportProcessor()


def test_case_6():
    int_0 = -29
    module_0.lazy_import(int_0, int_0, int_0)


def test_case_7():
    str_0 = "\rdI<H3Pmg~p"
    module_0.lazy_import(str_0, str_0)


def test_case_8():
    var_0 = module_0.disallow_proxying()


def test_case_9():
    import_processor_0 = module_0.disallow_proxying()
    module_0.ScopeReplacer(import_processor_0, import_processor_0, import_processor_0)


def test_case_10():
    import_processor_0 = module_0.ImportProcessor()
    import_processor_0.lazy_import(import_processor_0, import_processor_0)


def test_case_11():
    str_0 = "\rdI<H3Pmg~p"
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        str_0, str_0, str_0
    )
    var_0 = illegal_use_of_scope_replacer_0.__eq__(str_0)
    illegal_use_of_scope_replacer_0.__str__()


def test_case_12():
    str_0 = "~_}2Y\n\n]B_MZ"
    module_0.lazy_import(str_0, str_0)


def test_case_13():
    str_0 = "6v:|F}Y&j)\nzJ,k"
    dict_0 = {str_0: str_0}
    import_replacer_0 = module_0.ImportReplacer(dict_0, str_0, str_0)
    import_replacer_0.__getattribute__(dict_0)


def test_case_14():
    str_0 = "SB5J#NE(cQQqZ$ RP"
    module_0.lazy_import(str_0, str_0)


def test_case_15():
    none_type_0 = None
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        none_type_0, none_type_0
    )
    var_0 = illegal_use_of_scope_replacer_0.__eq__(illegal_use_of_scope_replacer_0)
    illegal_use_of_scope_replacer_0.__str__()


def test_case_16():
    str_0 = '~_}Y;(\n]B"M7'
    module_0.lazy_import(str_0, str_0, str_0)


def test_case_17():
    str_0 = "\n    Turn a function to a bound method on an instance\n\n    >>> class Foo(object):\n    ...     def __init__(self, x, y):\n    ...         self.x = x\n    ...         self.y = y\n    >>> foo = Foo(2, 3)\n    >>> my_unbound_method = lambda self: self.x * self.y\n    >>> bind(foo, my_unbound_method, 'multiply')\n    >>> foo.multiply()  # noinspection PyUnresolvedReferences\n    6\n\n    :param object instance: some object\n    :param callable func: unbound method (i.e. a function that takes `self` argument, that you now\n        want to be bound to this class as a method)\n    :param str as_name: name of the method to create on the object\n    "
    module_0.lazy_import(str_0, str_0, str_0)


def test_case_18():
    str_0 = " Merge multiple queues together\n\n    >>> q1, q2, q3 = [Queue() for _ in range(3)]\n    >>> out_q = merge(q1, q2, q3)\n    "
    module_0.lazy_import(str_0, str_0)


def test_case_19():
    str_0 = 'Pu~_;\n!]B"M7'
    dict_0 = {str_0: str_0}
    import_replacer_0 = module_0.ImportReplacer(dict_0, str_0, str_0)
    import_processor_0 = module_0.ImportProcessor()
    module_0.ImportReplacer(
        dict_0, dict_0, import_replacer_0, import_processor_0, dict_0
    )


def test_case_20():
    str_0 = 'Pu~_;\n!]B"M7'
    dict_0 = {str_0: str_0}
    import_replacer_0 = module_0.ImportReplacer(dict_0, str_0, str_0, dict_0)
    str_1 = "\x0c#zbPjWnD\t"
    module_0.lazy_import(import_replacer_0, str_1)


def test_case_21():
    str_0 = 'Pu~_;\n!]B"M7'
    dict_0 = {}
    import_replacer_0 = module_0.ImportReplacer(dict_0, str_0, dict_0, str_0)
    module_0.lazy_import(dict_0, import_replacer_0)
