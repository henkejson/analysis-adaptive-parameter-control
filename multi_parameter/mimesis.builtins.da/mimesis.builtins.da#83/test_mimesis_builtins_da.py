# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import mimesis.builtins.da as module_0


def test_case_0():
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider()
    str_0 = denmark_spec_provider_0.cpr()


def test_case_1():
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider()
    str_0 = denmark_spec_provider_0.cpr()
    denmark_spec_provider_1 = module_0.DenmarkSpecProvider(str_0)
    str_1 = denmark_spec_provider_0.cpr()
    str_2 = denmark_spec_provider_0.__str__()


def test_case_2():
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider()
    str_0 = denmark_spec_provider_0.cpr()
    denmark_spec_provider_1 = module_0.DenmarkSpecProvider(denmark_spec_provider_0)
    denmark_spec_provider_2 = module_0.DenmarkSpecProvider()
    str_1 = denmark_spec_provider_0.cpr()


def test_case_3():
    bytes_0 = b"\x9d}\x8bJ\xc3|\x96#\xe2\xeb\xd1e\xac\xdf\xd6\xb6"
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider(bytes_0)


def test_case_4():
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider()
    str_0 = denmark_spec_provider_0.cpr()
    denmark_spec_provider_1 = module_0.DenmarkSpecProvider(denmark_spec_provider_0)
    denmark_spec_provider_2 = module_0.DenmarkSpecProvider()
    str_1 = denmark_spec_provider_0.cpr()


def test_case_5():
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider()
    str_0 = denmark_spec_provider_0.cpr()


def test_case_6():
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider()
    denmark_spec_provider_1 = module_0.DenmarkSpecProvider()
    str_0 = denmark_spec_provider_0.cpr()
