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
    object_0 = module_0.disallow_proxying()
    var_0 = module_0.IllegalUseOfScopeReplacer(object_0, object_0)
    var_0.__unicode__()


def test_case_2():
    str_0 = "\n    Turn a function to a bound method on an instance\n\n    >>> class Foo(object):\n    ...     def __init__(self, xX y):\n    ...         self.x = x\n    ...         self.y = y\n    >>> foo = Foo(2, 3)\n    >>> my_unbound_method = lambda self: self.x * self.y\n    >> bind(foo, my_unbound_method, 'multiply')\n    >>> foo.multiply()  # noinspection PyUnresolved\x0ceferences\n    6\n\n   :param object insLance: some object\n   :param callable func: unbou+d method (i.e. a function that takes `self` argument, that you now\n        want to be bound to this clasl as a method)\n    :param str as_name: name of the method to create on the object\n    "
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(str_0, str_0)
    var_0 = illegal_use_of_scope_replacer_0.__eq__(str_0)
    module_0.lazy_import(str_0, str_0)


def test_case_3():
    float_0 = 857.52852
    module_0.ImportReplacer(float_0, float_0, float_0)


def test_case_4():
    import_processor_0 = module_0.ImportProcessor()


def test_case_5():
    bytes_0 = b"\xe8\xd5\xa4|\xed1\xe9\xb4\xf8\xb8\xf6\x13\xc7Y\xb3l=7\xb9"
    module_0.lazy_import(bytes_0, bytes_0, bytes_0)


def test_case_6():
    str_0 = 'a:{hvrHUU(nxGw^cn9"'
    module_0.lazy_import(str_0, str_0)


def test_case_7():
    str_0 = "X\x0bRguBlm'z8GW#>BlP"
    module_0.lazy_import(str_0, str_0)


def test_case_8():
    var_0 = module_0.disallow_proxying()


def test_case_9():
    var_0 = module_0.disallow_proxying()
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(var_0, var_0)
    illegal_use_of_scope_replacer_0.__repr__()


def test_case_10():
    str_0 = "Invalid pattern(s) found. %(msg)s"
    module_0.lazy_import(str_0, str_0)


def test_case_11():
    str_0 = "\n    Turn a function to a bound method on an instance\n\n    >>> class Foo(object):\n    ...     def __init__(self, xX y):\n    ...         self.x = x\n    ...         self.y = y\n    >>> foo = Foo(2, 3)\n    >>> my_unbound_method = lambda self: self.x * self.y\n    >>> bind(foo, my_unbound_method, 'multiply')\n    >>> foo.multiply()  # noinspection PyUnresolved\x0ceferences\n    6\n\n    :param object instance: some object\n    :param callable func: unbou+d method (i.e. a function that takes `self` argument, that you now\n        want to be bound to this class as a method)\n    :param str as_name: name of the method to create on the object\n    "
    module_0.lazy_import(str_0, str_0, str_0)


def test_case_12():
    str_0 = "Odz$l) e.-gj"
    dict_0 = {str_0: str_0, str_0: str_0}
    import_replacer_0 = module_0.ImportReplacer(dict_0, str_0, dict_0, dict_0)
    module_0.lazy_import(dict_0, import_replacer_0, str_0)


def test_case_13():
    var_0 = module_0.disallow_proxying()
    illegal_use_of_scope_replacer_0 = module_0.IllegalUseOfScopeReplacer(var_0, var_0)
    var_1 = illegal_use_of_scope_replacer_0.__eq__(illegal_use_of_scope_replacer_0)
    illegal_use_of_scope_replacer_0.__unicode__()


def test_case_14():
    str_0 = "k}2@+"
    bytes_0 = b"\x03\xb2\x95\\(\x89~"
    int_0 = 1482
    int_1 = -622
    module_0.ImportReplacer(bytes_0, int_0, str_0, int_1, int_1)


def test_case_15():
    str_0 = "#b>-o^b6#r\t/Mn!PO\x0c:D"
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0, str_0: str_0, str_0: str_0}
    import_replacer_0 = module_0.ImportReplacer(dict_0, str_0, dict_0, dict_0)
    module_0.lazy_import(import_replacer_0, str_0)


def test_case_16():
    str_0 = "\n    Turn a function to a bound method on an instance\n\n    >>> class Foo(object:\n    ...     def __init__(self, xX y):\n    ...         self.x = x\n    ...         self.y = y\n    >>> foo = Foo(2, 3)\n    >>> my_unbound_method = lambda self: self. * self.y\n    >> bind(foo, my_unbound_method, 'multiply')\n    >>> foo.multiply()  # noinspection PyUnresolved\x0ceferences\n    6\n\n   :param object insLance: some object\n   :param callable func: unbou+d method (i.e. a function that takes `self` argument, that you now\n        want to be bound to this clasl as a method)\n    :param str as_name: name of the method to create on the object\n    "
    dict_0 = {str_0: str_0, str_0: str_0}
    import_replacer_0 = module_0.ImportReplacer(dict_0, str_0, dict_0)
    module_0.lazy_import(import_replacer_0, import_replacer_0, dict_0)


def test_case_17():
    str_0 = "\n    Turn a function to a bound method on an instance\n\n    >>> class Foo(object:\n    ...     def __init__(self, xX y):\n    ...         self.x = x\n    ...         self.y = y\n    >>> foo = Foo(2, 3)\n    >>> my_unbound_method = lambda self: self. * self.y\n    >> bind(foo, my_unbound_method, 'multiply')\n    >>> foo.multiply()  # noinspection PyUnresolved\x0ceferences\n    6\n\n   :param object insLance: some object\n   :param callable func: unbou+d method (i.e. a function that takes `self` argument, that you now\n        want to be bound to this clasl as a method)\n    :param str as_name: name of the method to create on the object\n    "
    module_0.lazy_import(str_0, str_0)
