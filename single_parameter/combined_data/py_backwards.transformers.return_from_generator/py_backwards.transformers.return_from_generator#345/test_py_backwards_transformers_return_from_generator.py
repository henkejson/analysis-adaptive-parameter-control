# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.return_from_generator as module_1
import typed_ast.ast3 as module_2


def test_case_0():
    bin_op_0 = module_0.BinOp()
    return_from_generator_transformer_0 = module_1.ReturnFromGeneratorTransformer(
        bin_op_0
    )


def test_case_1():
    bin_op_0 = module_0.BinOp()
    var_0 = module_2.dump(bin_op_0)
    return_from_generator_transformer_0 = module_1.ReturnFromGeneratorTransformer(
        bin_op_0
    )
    var_1 = module_2.parse(var_0)
    function_def_0 = return_from_generator_transformer_0.visit_FunctionDef(var_1)


def test_case_2():
    bin_op_0 = module_0.BinOp()
    return_from_generator_transformer_0 = module_1.ReturnFromGeneratorTransformer(
        bin_op_0
    )
    var_0 = module_2.iter_fields(bin_op_0)
    list_0 = [var_0, return_from_generator_transformer_0, var_0]
    function_def_0 = module_0.FunctionDef(*list_0)
    function_def_1 = return_from_generator_transformer_0.visit_FunctionDef(
        function_def_0
    )
