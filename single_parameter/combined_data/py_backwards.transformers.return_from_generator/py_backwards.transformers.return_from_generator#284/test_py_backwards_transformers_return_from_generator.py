# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.return_from_generator as module_1
import typed_ast.ast3 as module_2


def test_case_0():
    expr_context_0 = module_0.expr_context()
    return_from_generator_transformer_0 = module_1.ReturnFromGeneratorTransformer(
        expr_context_0
    )


def test_case_1():
    bytes_0 = b""
    return_from_generator_transformer_0 = module_1.ReturnFromGeneratorTransformer(
        bytes_0
    )
    var_0 = module_2.parse(bytes_0)
    function_def_0 = return_from_generator_transformer_0.visit_FunctionDef(var_0)


def test_case_2():
    bytes_0 = b"F"
    var_0 = module_2.parse(bytes_0)
    return_from_generator_transformer_0 = module_1.ReturnFromGeneratorTransformer(var_0)
    function_def_0 = return_from_generator_transformer_0.visit_FunctionDef(var_0)
