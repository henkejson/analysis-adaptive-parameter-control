# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.return_from_generator as module_1


def test_case_0():
    arguments_0 = module_0.arguments()
    return_from_generator_transformer_0 = module_1.ReturnFromGeneratorTransformer(
        arguments_0
    )


def test_case_1():
    str_0 = "]Nzcg'HT"
    list_0 = [str_0, str_0, str_0]
    function_def_0 = module_0.FunctionDef(*list_0)
    return_from_generator_transformer_0 = module_1.ReturnFromGeneratorTransformer(str_0)
    function_def_1 = return_from_generator_transformer_0.visit_FunctionDef(
        function_def_0
    )


def test_case_2():
    str_0 = ""
    list_0 = [str_0, str_0, str_0]
    function_def_0 = module_0.FunctionDef(*list_0)
    return_from_generator_transformer_0 = module_1.ReturnFromGeneratorTransformer(
        list_0
    )
    return_from_generator_transformer_1 = module_1.ReturnFromGeneratorTransformer(
        function_def_0
    )
    if_0 = module_0.If()
    return_from_generator_transformer_2 = module_1.ReturnFromGeneratorTransformer(if_0)
    function_def_1 = return_from_generator_transformer_1.visit_FunctionDef(
        function_def_0
    )
    return_from_generator_transformer_3 = module_1.ReturnFromGeneratorTransformer(
        function_def_0
    )
    return_from_generator_transformer_4 = module_1.ReturnFromGeneratorTransformer(
        return_from_generator_transformer_1
    )
    return_from_generator_transformer_5 = module_1.ReturnFromGeneratorTransformer(
        function_def_0
    )
    return_from_generator_transformer_6 = module_1.ReturnFromGeneratorTransformer(
        function_def_0
    )
    function_def_2 = return_from_generator_transformer_6.visit_FunctionDef(
        function_def_1
    )
    return_from_generator_transformer_7 = module_1.ReturnFromGeneratorTransformer(
        function_def_0
    )
