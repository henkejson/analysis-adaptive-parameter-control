# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import mimesis.builtins.da as module_0
import mimesis.providers.base as module_1


def test_case_0():
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider()
    str_0 = denmark_spec_provider_0.cpr()
    str_1 = denmark_spec_provider_0.__str__()
    denmark_spec_provider_1 = module_0.DenmarkSpecProvider()


def test_case_1():
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider()
    str_0 = denmark_spec_provider_0.cpr()
    denmark_spec_provider_0.validate_enum(str_0, str_0)


def test_case_2():
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider()
    str_0 = denmark_spec_provider_0.__str__()
    denmark_spec_provider_1 = module_0.DenmarkSpecProvider(str_0)
    str_1 = denmark_spec_provider_1.__str__()
    str_2 = denmark_spec_provider_0.cpr()
    str_3 = denmark_spec_provider_1.get_current_locale()


def test_case_3():
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider()
    str_0 = denmark_spec_provider_0.cpr()
    str_1 = denmark_spec_provider_0.cpr()
    str_2 = denmark_spec_provider_0.cpr()


def test_case_4():
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider()
    str_0 = denmark_spec_provider_0.cpr()
    denmark_spec_provider_0.validate_enum(denmark_spec_provider_0, str_0)


def test_case_5():
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider()
    str_0 = denmark_spec_provider_0.cpr()
    none_type_0 = None
    tuple_0 = ()
    base_provider_0 = module_1.BaseProvider(seed=tuple_0)
    base_provider_0.validate_enum(none_type_0, none_type_0)
