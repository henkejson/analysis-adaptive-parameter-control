# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import mimesis.builtins.da as module_0


def test_case_0():
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider()
    str_0 = denmark_spec_provider_0.cpr()


def test_case_1():
    bytes_0 = b"\xad\x97\xda\xda\xdc-\\\xd3|"
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider(bytes_0)
    str_0 = denmark_spec_provider_0.cpr()
    str_1 = denmark_spec_provider_0.cpr()
    str_2 = denmark_spec_provider_0.cpr()
    str_3 = denmark_spec_provider_0.cpr()
    str_4 = "iSI\r"
    denmark_spec_provider_1 = module_0.DenmarkSpecProvider(str_4)


def test_case_2():
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider()
    str_0 = denmark_spec_provider_0.cpr()
    str_1 = denmark_spec_provider_0.cpr()


def test_case_3():
    bool_0 = False
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider(bool_0)


def test_case_4():
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider()
    str_0 = denmark_spec_provider_0.cpr()
    str_1 = denmark_spec_provider_0.cpr()


def test_case_5():
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider()
    str_0 = denmark_spec_provider_0.cpr()


def test_case_6():
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider()
    str_0 = denmark_spec_provider_0.cpr()
