# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typesystem.tokenize.positional_validation as module_0
import typesystem.schemas as module_1


def test_case_0():
    bool_0 = True
    module_0.validate_with_positions(token=bool_0, validator=bool_0)


def test_case_1():
    dict_0 = {}
    schema_0 = module_1.Schema(dict_0)
    validation_result_0 = schema_0.validate_or_error(schema_0)
    module_0.validate_with_positions(token=validation_result_0, validator=schema_0)
