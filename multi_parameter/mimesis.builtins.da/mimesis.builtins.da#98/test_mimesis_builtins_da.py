# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import mimesis.builtins.da as module_0
import mimesis.enums as module_1


def test_case_0():
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider()
    str_0 = denmark_spec_provider_0.cpr()


def test_case_1():
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider()
    str_0 = denmark_spec_provider_0.cpr()


def test_case_2():
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider()
    locale_0 = module_1.Locale.JA
    generator_0 = denmark_spec_provider_0.override_locale(locale_0)
    str_0 = denmark_spec_provider_0.cpr()


def test_case_3():
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider()


def test_case_4():
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider()
    none_type_0 = None
    generator_0 = denmark_spec_provider_0.override_locale(none_type_0)
    str_0 = denmark_spec_provider_0.cpr()


def test_case_5():
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider()
    none_type_0 = None
    denmark_spec_provider_1 = module_0.DenmarkSpecProvider(none_type_0)
    str_0 = denmark_spec_provider_1.cpr()
