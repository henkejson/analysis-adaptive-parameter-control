# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import mimesis.builtins.da as module_0


def test_case_0():
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider()
    str_0 = denmark_spec_provider_0.cpr()


def test_case_1():
    none_type_0 = None
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider()
    denmark_spec_provider_1 = module_0.DenmarkSpecProvider(none_type_0)
    denmark_spec_provider_2 = module_0.DenmarkSpecProvider()
    str_0 = denmark_spec_provider_1.cpr()
    dict_0 = {}
    none_type_1 = denmark_spec_provider_2.update_dataset(dict_0)
    str_1 = denmark_spec_provider_0.cpr()
    denmark_spec_provider_3 = module_0.DenmarkSpecProvider()
    denmark_spec_provider_4 = module_0.DenmarkSpecProvider()
    str_2 = denmark_spec_provider_0.cpr()


def test_case_2():
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider()
    str_0 = denmark_spec_provider_0.cpr()


def test_case_3():
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider()


def test_case_4():
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider()
    str_0 = denmark_spec_provider_0.cpr()


def test_case_5():
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider()
    str_0 = denmark_spec_provider_0.cpr()
