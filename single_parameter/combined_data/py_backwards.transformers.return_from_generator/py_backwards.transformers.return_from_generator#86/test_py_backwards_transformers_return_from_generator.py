# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import py_backwards.transformers.return_from_generator as module_0
import typed_ast._ast3 as module_1


def test_case_0():
    none_type_0 = None
    return_from_generator_transformer_0 = module_0.ReturnFromGeneratorTransformer(
        none_type_0
    )


def test_case_1():
    list_0 = []
    return_from_generator_transformer_0 = module_0.ReturnFromGeneratorTransformer(
        list_0
    )
    list_1 = [
        list_0,
        return_from_generator_transformer_0,
        list_0,
        return_from_generator_transformer_0,
    ]
    async_for_0 = module_1.AsyncFor(*list_1)
    function_def_0 = return_from_generator_transformer_0.visit_FunctionDef(async_for_0)
