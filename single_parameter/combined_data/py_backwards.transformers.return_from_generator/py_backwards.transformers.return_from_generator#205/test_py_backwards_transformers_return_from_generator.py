# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.return_from_generator as module_1
import typed_ast.ast3 as module_2


def test_case_0():
    tuple_0 = module_0.Tuple()
    return_from_generator_transformer_0 = module_1.ReturnFromGeneratorTransformer(
        tuple_0
    )


def test_case_1():
    str_0 = "Zed/tWX"
    return_from_generator_transformer_0 = module_1.ReturnFromGeneratorTransformer(str_0)
    var_0 = module_2.parse(str_0)
    function_def_0 = return_from_generator_transformer_0.visit_FunctionDef(var_0)


def test_case_2():
    is_0 = module_0.Is()
    return_from_generator_transformer_0 = module_1.ReturnFromGeneratorTransformer(is_0)
    str_0 = ""
    return_from_generator_transformer_1 = module_1.ReturnFromGeneratorTransformer(str_0)
    var_0 = module_2.parse(str_0)
    function_def_0 = return_from_generator_transformer_1.visit_FunctionDef(var_0)
    function_def_1 = module_2.fix_missing_locations(var_0)
