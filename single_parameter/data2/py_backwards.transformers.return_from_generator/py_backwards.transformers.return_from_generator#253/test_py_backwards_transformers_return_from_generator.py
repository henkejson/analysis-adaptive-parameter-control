# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.return_from_generator as module_1


def test_case_0():
    mult_0 = module_0.Mult()
    return_from_generator_transformer_0 = module_1.ReturnFromGeneratorTransformer(
        mult_0
    )
