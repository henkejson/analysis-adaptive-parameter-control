# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.return_from_generator as module_1
import typed_ast.ast3 as module_2


def test_case_0():
    starred_0 = module_0.Starred()
    return_from_generator_transformer_0 = module_1.ReturnFromGeneratorTransformer(
        starred_0
    )


def test_case_1():
    str_0 = "{;T"
    return_from_generator_transformer_0 = module_1.ReturnFromGeneratorTransformer(str_0)
    list_0 = [str_0, return_from_generator_transformer_0, str_0, str_0, str_0]
    function_def_0 = module_0.FunctionDef(*list_0)
    function_def_1 = return_from_generator_transformer_0.visit_FunctionDef(
        function_def_0
    )


def test_case_2():
    str_0 = ""
    function_type_0 = module_0.FunctionType()
    return_from_generator_transformer_0 = module_1.ReturnFromGeneratorTransformer(
        function_type_0
    )
    return_from_generator_transformer_1 = module_1.ReturnFromGeneratorTransformer(
        function_type_0
    )
    return_from_generator_transformer_2 = module_1.ReturnFromGeneratorTransformer(
        function_type_0
    )
    list_0 = [str_0, return_from_generator_transformer_0, str_0]
    function_def_0 = module_0.FunctionDef(*list_0)
    function_def_1 = return_from_generator_transformer_0.visit_FunctionDef(
        function_def_0
    )
    module_2.literal_eval(function_def_0)


def test_case_3():
    str_0 = "D_"
    function_type_0 = module_0.FunctionType()
    return_from_generator_transformer_0 = module_1.ReturnFromGeneratorTransformer(
        function_type_0
    )
    return_from_generator_transformer_1 = module_1.ReturnFromGeneratorTransformer(
        function_type_0
    )
    var_0 = module_2.iter_child_nodes(return_from_generator_transformer_1)
    list_0 = [str_0, return_from_generator_transformer_0, str_0]
    function_def_0 = module_0.FunctionDef(*list_0)
    function_def_1 = return_from_generator_transformer_0.visit_FunctionDef(
        function_def_0
    )
    var_1 = module_2.parse(str_0)
    function_def_2 = return_from_generator_transformer_0.visit_FunctionDef(var_1)
    module_2.dump(var_0)
