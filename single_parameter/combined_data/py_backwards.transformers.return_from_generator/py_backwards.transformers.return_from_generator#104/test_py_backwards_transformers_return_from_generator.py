# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import py_backwards.transformers.return_from_generator as module_0
import typed_ast.ast3 as module_1


def test_case_0():
    none_type_0 = None
    return_from_generator_transformer_0 = module_0.ReturnFromGeneratorTransformer(
        none_type_0
    )


def test_case_1():
    str_0 = "_wWminreg"
    return_from_generator_transformer_0 = module_0.ReturnFromGeneratorTransformer(str_0)
    var_0 = module_1.parse(str_0)
    function_def_0 = return_from_generator_transformer_0.visit_FunctionDef(var_0)


def test_case_2():
    str_0 = ""
    return_from_generator_transformer_0 = module_0.ReturnFromGeneratorTransformer(str_0)
    var_0 = module_1.parse(str_0)
    function_def_0 = return_from_generator_transformer_0.visit_FunctionDef(var_0)
