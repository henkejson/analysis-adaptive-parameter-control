# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import mimesis.builtins.da as module_0
import mimesis.providers.base as module_1


def test_case_0():
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider()
    generator_0 = denmark_spec_provider_0.override_locale(denmark_spec_provider_0)
    str_0 = denmark_spec_provider_0.cpr()
    str_1 = denmark_spec_provider_0.cpr()


def test_case_1():
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider()
    generator_0 = denmark_spec_provider_0.override_locale(denmark_spec_provider_0)
    str_0 = denmark_spec_provider_0.cpr()
    str_1 = denmark_spec_provider_0.cpr()
    str_2 = denmark_spec_provider_0.cpr()


def test_case_2():
    none_type_0 = None
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider(none_type_0)
    denmark_spec_provider_1 = module_0.DenmarkSpecProvider()
    list_0 = []
    str_0 = denmark_spec_provider_1.cpr()
    base_provider_0 = module_1.BaseProvider()
    denmark_spec_provider_2 = module_0.DenmarkSpecProvider()
    base_provider_0.validate_enum(list_0, list_0)


def test_case_3():
    none_type_0 = None
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider(none_type_0)
    str_0 = denmark_spec_provider_0.get_current_locale()
    generator_0 = denmark_spec_provider_0.override_locale(none_type_0)


def test_case_4():
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider()
    generator_0 = denmark_spec_provider_0.override_locale(denmark_spec_provider_0)
    str_0 = denmark_spec_provider_0.cpr()
    str_1 = denmark_spec_provider_0.cpr()
    str_2 = denmark_spec_provider_0.get_current_locale()
