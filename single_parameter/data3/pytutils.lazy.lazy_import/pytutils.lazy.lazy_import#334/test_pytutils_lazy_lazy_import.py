# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pytutils.lazy.lazy_import as module_0


def test_case_0():
    bool_0 = True
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        bool_0, bool_0, bool_0
    )
    illegal_use_of_scope_replacer_0.__unicode__()


def test_case_1():
    bool_0 = False
    none_type_0 = None
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        bool_0, bool_0, none_type_0
    )


def test_case_2():
    bool_0 = False
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        bool_0, bool_0, bool_0
    )
    illegal_use_of_scope_replacer_0.__unicode__()


def test_case_3():
    str_0 = "\n    Turn a function to a bound method on an instance\n\n    >>> class Foo(object):\n    ...     def __init__(self,x, y):\n    ...         self.x = x\n    ...         self.y = y\n    >>> foo = Foo(2, 3)\n    >>> my_unbound_method = lambda self: self.x * self.y\n    >>> bind(foo, my_unbound_method, 'multiply')\n    >>> foo.multiply()  # noinspection PyUnresolvedReferences\n    6\n\n    :param object instance: some object\n    :param callable func: unbound method (i.e. a function tha takes `self` argument, that you now\n        want to be bound to this class as a method)\n    :param str as_name:name of the method tocreate on the object\n    "
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(str_0, str_0)
    var_0 = illegal_use_of_scope_replacer_0.__eq__(str_0)
    illegal_use_of_scope_replacer_0.__str__()


def test_case_4():
    bool_0 = True
    list_0 = [bool_0, bool_0]
    import_replacer_0 = module_0.ImportReplacer(list_0, bool_0, list_0)
    import_replacer_0.__getattribute__(bool_0)


def test_case_5():
    str_0 = "\n            Override the __mro__ to fool `isinstance`.\n            "
    import_processor_0 = module_0.ImportProcessor()
    dict_0 = {
        str_0: import_processor_0,
        import_processor_0: str_0,
        str_0: import_processor_0,
    }
    import_replacer_0 = module_0.ImportReplacer(
        dict_0, str_0, str_0, children=import_processor_0
    )
    none_type_0 = None
    import_replacer_1 = module_0.ImportReplacer(
        dict_0, none_type_0, import_processor_0, dict_0
    )
    module_0.lazy_import(import_replacer_1, str_0)


def test_case_6():
    import_processor_0 = module_0.ImportProcessor()


def test_case_7():
    str_0 = "\n    Ensure string is decoded (eg unicode); convert using specified parameters if we have to.\n\n    :param str|bytes|bytesarray|memoryview s: string/bytes\n    :param str encoding: Decode using this encoding\n    :param str errors: How to handle errors\n    :return bytes|bytesarray|memoryview: Decoded string as bytes\n\n    :return: Encoded string\n    :rtype: bytes\n    "
    module_0.lazy_import(str_0, str_0, str_0)


def test_case_8():
    str_0 = "\n    Turn a function to a bound method on an instance\n\n    >>> class Foo(object):\n    ...     def __init__(self, x, y):\n    ...         self.x = x\n    ...         self.y = y\n    >>> foo = Foo(2, 3)\n    >>> my_unbound_method = lambda self: self.x * self.y\n    >>> bind(foo, my_unbound_method, 'multiply')\n    >>> foo.multiply()  # noinspection PyUnresolvedReferences\n    6\n\n    :param object instance: some object\n    :param callable func: unbound method (i.e. a function tha takes `self` argument, that you now\n        want to be bound to this class as a method)\n    :param str as_name: name of the method tocreate on the object\n    "
    module_0.lazy_import(str_0, str_0)


def test_case_9():
    bool_0 = False
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        bool_0, bool_0, bool_0
    )
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_10():
    import_processor_0 = module_0.disallow_proxying()


def test_case_11():
    int_0 = 776
    module_0.ImportReplacer(int_0, int_0, int_0, int_0, int_0)


def test_case_12():
    str_0 = "kwT`<#Vw`?;"
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0}
    import_replacer_0 = module_0.ImportReplacer(dict_0, str_0, str_0, children=dict_0)
    module_0.lazy_import(dict_0, import_replacer_0)


def test_case_13():
    str_0 = ""
    module_0.lazy_import(str_0, str_0, str_0)


def test_case_14():
    str_0 = "\n    Turn a function to a bound method on an instance\n\n    >>> class Foo(object):\n    ...    def __init__(self, x, y):\n    ...         self.x = x\n    ...         self.y = y\n    >>> foo = Foo(2, 3)\n    >>> my_unbound_method = lambda self: self.x * self.y\n    >>> bind(foo my_unbound_method, 'multiply')\n    >>> foo.multiply()  # noinspection PyUnresolvedReferences\n    6\n\n    :param object instance: some object\n    :param callable func: unbound method (i.e. a function that takes `self` argument, that you now\n        want to be bound to this class as a method\n    :param str as_name: name of the method to create n the object\n    "
    module_0.lazy_import(str_0, str_0)


def test_case_15():
    bool_0 = False
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(
        bool_0, bool_0, bool_0
    )
    var_0 = illegal_use_of_scope_replacer_0.__eq__(illegal_use_of_scope_replacer_0)
    illegal_use_of_scope_replacer_0.__unicode__()


def test_case_16():
    str_0 = "\n    Turn a function to a bound method on an instance\n\n    >>> class Foo(object:\n    ...     def __init__(self, x, y):\n    ...         self.x = x\n    ...         self.y = y\n    >>> foo = Foo(2, 3)\n    >>> my_unbound_method = lambda self: self.x * self.y\n    >>> bind(foo, my_unbound_method, 'multiply')\n    >>> foo.multiply()  # noinspection PyUnresolvedReferences\n    6\n\n    :param object instance: some object\n    :param callable func: unbound method (i.e. a function tha takes `self` argument, that you now\n        want to be bound to this class a a method)\n    :param str as_name: name of the method tocreate on the object\n    "
    module_0.lazy_import(str_0, str_0)


def test_case_17():
    str_0 = "Convert the given text into a bunch of lazy import objects.\n\n        This takes a text string, which should be similar to normal python\n        import markup.\n        "
    dict_0 = {}
    import_replacer_0 = module_0.ImportReplacer(dict_0, str_0, dict_0, str_0)
    import_replacer_0.__setattr__(str_0, dict_0)
