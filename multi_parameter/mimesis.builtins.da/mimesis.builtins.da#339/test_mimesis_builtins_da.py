# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import mimesis.builtins.da as module_0


def test_case_0():
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider()
    str_0 = denmark_spec_provider_0.cpr()


def test_case_1():
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider()
    str_0 = denmark_spec_provider_0.cpr()
    str_1 = ";Mj5? bYgDL0WhU"
    denmark_spec_provider_0.formatted_datetime(str_1)


def test_case_2():
    none_type_0 = None
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider(none_type_0)
    str_0 = denmark_spec_provider_0.cpr()


def test_case_3():
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider()
    denmark_spec_provider_1 = module_0.DenmarkSpecProvider()


def test_case_4():
    none_type_0 = None
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider(none_type_0)
    str_0 = denmark_spec_provider_0.cpr()


def test_case_5():
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider()
    str_0 = denmark_spec_provider_0.cpr()
    str_1 = denmark_spec_provider_0.__str__()


def test_case_6():
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider()
    str_0 = denmark_spec_provider_0.cpr()
    str_1 = denmark_spec_provider_0.__str__()
