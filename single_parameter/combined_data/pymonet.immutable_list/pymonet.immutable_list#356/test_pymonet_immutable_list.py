# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pymonet.immutable_list as module_0


def test_case_0():
    immutable_list_0 = module_0.ImmutableList()
    bool_0 = immutable_list_0.__eq__(immutable_list_0)
    immutable_list_0.filter(immutable_list_0)


def test_case_1():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.__len__()
    immutable_list_1 = immutable_list_0.unshift(var_0)
    var_1 = immutable_list_0.find(var_0)
    bool_0 = immutable_list_0.__eq__(var_1)
    immutable_list_1.find(var_1)


def test_case_2():
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.append(none_type_0)
    immutable_list_1.filter(none_type_0)


def test_case_3():
    bool_0 = False
    immutable_list_0 = module_0.ImmutableList(is_empty=bool_0)
    immutable_list_1 = immutable_list_0.unshift(bool_0)
    var_0 = immutable_list_0.to_list()
    immutable_list_2 = var_0.append(immutable_list_1)
    immutable_list_1.__add__(var_0)


def test_case_4():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.find(immutable_list_0)
    bool_0 = immutable_list_0.__len__()
    immutable_list_0.filter(immutable_list_0)


def test_case_5():
    bytes_0 = b'\xd3i\x96"\x0e\x95\xc6\xa5\x1f\x9fY\x9ea\xc3x\xd9\xfb\xf9'
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(bytes_0)
    var_0 = immutable_list_1.__len__()
    immutable_list_1.find(immutable_list_0)


def test_case_6():
    immutable_list_0 = module_0.ImmutableList()
    bool_0 = immutable_list_0.to_list()
    immutable_list_0.filter(immutable_list_0)


def test_case_7():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = module_0.ImmutableList()
    immutable_list_2 = immutable_list_1.unshift(immutable_list_1)
    immutable_list_3 = immutable_list_2.append(immutable_list_1)
    immutable_list_4 = immutable_list_2.unshift(immutable_list_2)
    immutable_list_5 = module_0.ImmutableList(
        tail=immutable_list_1, is_empty=immutable_list_1
    )
    var_0 = immutable_list_5.to_list()


def test_case_8():
    immutable_list_0 = module_0.ImmutableList()
    none_type_0 = None
    immutable_list_0.map(none_type_0)


def test_case_9():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    var_0 = immutable_list_1.to_list()
    var_1 = immutable_list_0.find(var_0)
    immutable_list_2 = immutable_list_0.__add__(immutable_list_0)
    str_0 = var_0.__str__()
    bool_0 = True
    immutable_list_3 = module_0.ImmutableList(is_empty=bool_0)
    bool_1 = immutable_list_3.__eq__(immutable_list_1)
    immutable_list_1.map(immutable_list_1)


def test_case_10():
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_0.filter(immutable_list_0)


def test_case_11():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.find(immutable_list_0)
    immutable_list_0.filter(immutable_list_0)


def test_case_12():
    bool_0 = True
    none_type_0 = None
    immutable_list_0 = module_0.ImmutableList()
    immutable_list_1 = immutable_list_0.unshift(bool_0)
    immutable_list_1.find(none_type_0)


def test_case_13():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.find(immutable_list_0)
    var_1 = immutable_list_0.reduce(var_0, var_0)
    immutable_list_0.filter(immutable_list_0)


def test_case_14():
    bool_0 = False
    immutable_list_0 = module_0.ImmutableList(is_empty=bool_0)
    var_0 = immutable_list_0.to_list()
    var_1 = immutable_list_0.__len__()
    var_2 = immutable_list_0.reduce(var_0, var_0)
    immutable_list_1 = module_0.ImmutableList(var_2)
    immutable_list_2 = immutable_list_0.unshift(bool_0)
    immutable_list_2.reduce(immutable_list_0, immutable_list_0)


def test_case_15():
    immutable_list_0 = module_0.ImmutableList()


def test_case_16():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.find(immutable_list_0)
    immutable_list_1 = immutable_list_0.append(var_0)
    immutable_list_2 = immutable_list_1.__add__(immutable_list_0)
    immutable_list_3 = module_0.ImmutableList(
        immutable_list_2, is_empty=immutable_list_1
    )
    str_0 = immutable_list_1.__str__()
    immutable_list_1.filter(var_0)


def test_case_17():
    none_type_0 = None
    none_type_1 = None
    bool_0 = False
    immutable_list_0 = module_0.ImmutableList(none_type_1, none_type_1, bool_0)
    immutable_list_1 = immutable_list_0.unshift(none_type_0)


def test_case_18():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.__len__()
    var_1 = immutable_list_0.find(var_0)
    var_2 = module_0.ImmutableList(tail=var_0)
    bool_0 = immutable_list_0.__eq__(var_2)
    var_3 = immutable_list_0.reduce(var_2, var_2)
    immutable_list_0.filter(immutable_list_0)


def test_case_19():
    str_0 = "~DP$~5"
    immutable_list_0 = module_0.ImmutableList(str_0)
    immutable_list_1 = immutable_list_0.unshift(immutable_list_0)
    immutable_list_0.find(immutable_list_1)


def test_case_20():
    str_0 = "Z2Vczzb>"
    immutable_list_0 = module_0.ImmutableList(str_0)
    immutable_list_1 = immutable_list_0.__len__()
    immutable_list_0.find(immutable_list_0)


def test_case_21():
    immutable_list_0 = module_0.ImmutableList()
    var_0 = immutable_list_0.__len__()
    var_1 = immutable_list_0.find(var_0)
    var_2 = module_0.ImmutableList(immutable_list_0)
    var_3 = immutable_list_0.find(var_0)
    str_0 = "4`W"
    bool_0 = immutable_list_0.__eq__(str_0)
    var_2.reduce(var_3, str_0)
