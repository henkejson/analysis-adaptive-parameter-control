# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typesystem.tokenize.positional_validation as module_0
import typesystem.base as module_1


def test_case_0():
    bool_0 = True
    module_0.validate_with_positions(token=bool_0, validator=bool_0)


def test_case_1():
    none_type_0 = None
    module_1.BaseError(code=none_type_0, position=none_type_0)
