# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import mimesis.builtins.da as module_0
import mimesis.enums as module_1


def test_case_0():
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider()
    str_0 = denmark_spec_provider_0.cpr()


def test_case_1():
    int_0 = 875
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider(int_0)
    denmark_spec_provider_1 = module_0.DenmarkSpecProvider()
    denmark_spec_provider_2 = module_0.DenmarkSpecProvider(denmark_spec_provider_0)
    denmark_spec_provider_3 = module_0.DenmarkSpecProvider()
    locale_0 = module_1.Locale.DE_AT
    str_0 = denmark_spec_provider_3.cpr()
    none_type_0 = denmark_spec_provider_1.reseed()
    str_1 = denmark_spec_provider_0.__str__()
    str_2 = denmark_spec_provider_3.cpr()
    generator_0 = denmark_spec_provider_1.override_locale(locale_0)


def test_case_2():
    none_type_0 = None
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider(none_type_0)
    denmark_spec_provider_1 = module_0.DenmarkSpecProvider(denmark_spec_provider_0)
    str_0 = denmark_spec_provider_1.cpr()
    denmark_spec_provider_2 = module_0.DenmarkSpecProvider()
    tuple_0 = ()
    str_1 = denmark_spec_provider_2.get_current_locale()
    str_2 = denmark_spec_provider_2.cpr()
    denmark_spec_provider_3 = module_0.DenmarkSpecProvider()
    denmark_spec_provider_4 = module_0.DenmarkSpecProvider(tuple_0)
    str_3 = denmark_spec_provider_3.cpr()


def test_case_3():
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider()


def test_case_4():
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider()
    str_0 = denmark_spec_provider_0.cpr()


def test_case_5():
    none_type_0 = None
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider(none_type_0)
    str_0 = denmark_spec_provider_0.cpr()


def test_case_6():
    bytes_0 = b"]f\xe2\x94"
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider(bytes_0)
    str_0 = denmark_spec_provider_0.cpr()
    str_1 = denmark_spec_provider_0.cpr()
