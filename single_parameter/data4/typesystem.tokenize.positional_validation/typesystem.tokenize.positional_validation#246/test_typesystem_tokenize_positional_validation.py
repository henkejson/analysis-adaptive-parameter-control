# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typesystem.tokenize.positional_validation as module_0


def test_case_0():
    bytes_0 = b"J\xc9'\xb4"
    module_0.validate_with_positions(token=bytes_0, validator=bytes_0)
