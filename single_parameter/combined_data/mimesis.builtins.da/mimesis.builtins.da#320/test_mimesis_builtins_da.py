# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import mimesis.builtins.da as module_0


def test_case_0():
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider()
    str_0 = denmark_spec_provider_0.cpr()


def test_case_1():
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider()
    str_0 = denmark_spec_provider_0.cpr()
    denmark_spec_provider_1 = module_0.DenmarkSpecProvider()
    str_1 = denmark_spec_provider_1.cpr()
    str_2 = denmark_spec_provider_0.__str__()
    denmark_spec_provider_0.update_dataset(str_1)


def test_case_2():
    bytes_0 = b"A \tI\x89\xe5\t{C3\x15\x92\xbe\xc1/"
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider(bytes_0)
    str_0 = denmark_spec_provider_0.cpr()


def test_case_3():
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider()


def test_case_4():
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider()
    str_0 = denmark_spec_provider_0.cpr()
    str_1 = denmark_spec_provider_0.cpr()


def test_case_5():
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider()
    str_0 = denmark_spec_provider_0.cpr()
    denmark_spec_provider_0.validate_enum(str_0, denmark_spec_provider_0)
