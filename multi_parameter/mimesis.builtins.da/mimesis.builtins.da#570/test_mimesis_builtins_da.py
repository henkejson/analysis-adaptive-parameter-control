# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import mimesis.builtins.da as module_0


def test_case_0():
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider()
    str_0 = denmark_spec_provider_0.cpr()
    denmark_spec_provider_0.validate_enum(denmark_spec_provider_0, str_0)


def test_case_1():
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider()
    denmark_spec_provider_1 = module_0.DenmarkSpecProvider()
    str_0 = denmark_spec_provider_0.cpr()
    denmark_spec_provider_2 = module_0.DenmarkSpecProvider()
    denmark_spec_provider_3 = module_0.DenmarkSpecProvider()


def test_case_2():
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider()
    str_0 = denmark_spec_provider_0.cpr()
    denmark_spec_provider_0.validate_enum(
        denmark_spec_provider_0, denmark_spec_provider_0
    )


def test_case_3():
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider()


def test_case_4():
    bool_0 = True
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider()
    str_0 = denmark_spec_provider_0.cpr()
    str_1 = denmark_spec_provider_0.get_current_locale()
    denmark_spec_provider_1 = module_0.DenmarkSpecProvider(bool_0)


def test_case_5():
    bytes_0 = b"l\xa7\xa0\xbc\x03P\xf4\xe1\xd3\x86\xd3E\x16X\x1ey\xc8\xbbB"
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider(bytes_0)
    str_0 = denmark_spec_provider_0.cpr()
    denmark_spec_provider_1 = module_0.DenmarkSpecProvider()
    str_1 = denmark_spec_provider_1.cpr()


def test_case_6():
    denmark_spec_provider_0 = module_0.DenmarkSpecProvider()
    str_0 = denmark_spec_provider_0.cpr()
    denmark_spec_provider_0.validate_enum(denmark_spec_provider_0, str_0)
