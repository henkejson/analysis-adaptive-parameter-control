# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.return_from_generator as module_1
import typed_ast.ast3 as module_2


def test_case_0():
    aug_load_0 = module_0.AugLoad()
    return_from_generator_transformer_0 = module_1.ReturnFromGeneratorTransformer(
        aug_load_0
    )


def test_case_1():
    str_0 = "shlex_quote"
    var_0 = module_2.parse(str_0)
    return_from_generator_transformer_0 = module_1.ReturnFromGeneratorTransformer(str_0)
    function_def_0 = return_from_generator_transformer_0.visit_FunctionDef(var_0)


def test_case_2():
    dict_0 = {}
    list_0 = [dict_0, dict_0, dict_0, dict_0, dict_0]
    function_def_0 = module_0.FunctionDef(*list_0)
    return_from_generator_transformer_0 = module_1.ReturnFromGeneratorTransformer(
        function_def_0
    )
    function_def_1 = return_from_generator_transformer_0.visit_FunctionDef(
        function_def_0
    )


def test_case_3():
    str_0 = "shlex_quote"
    var_0 = module_2.parse(str_0)
    list_0 = [var_0, var_0, var_0, var_0, var_0, var_0]
    list_1 = [list_0, var_0, list_0, var_0]
    function_def_0 = module_0.FunctionDef(*list_1)
    return_from_generator_transformer_0 = module_1.ReturnFromGeneratorTransformer(
        list_1
    )
    function_def_1 = return_from_generator_transformer_0.visit_FunctionDef(
        function_def_0
    )


def test_case_4():
    dict_0 = {}
    var_0 = module_0.Return(**dict_0)
    return_from_generator_transformer_0 = module_1.ReturnFromGeneratorTransformer(var_0)
    dict_1 = module_2.iter_child_nodes(var_0)
    list_0 = [var_0, var_0, var_0, var_0, var_0, var_0]
    list_1 = [list_0, dict_1, list_0, dict_1]
    return_from_generator_transformer_1 = module_1.ReturnFromGeneratorTransformer(var_0)
    return_from_generator_transformer_2 = return_from_generator_transformer_1.visit(
        var_0
    )
    var_1 = module_2.iter_fields(var_0)
    function_def_0 = module_0.FunctionDef(*list_1)
    return_from_generator_transformer_3 = module_1.ReturnFromGeneratorTransformer(
        list_1
    )
    return_from_generator_transformer_3.visit_FunctionDef(function_def_0)


def test_case_5():
    var_0 = module_0.YieldFrom()
    list_0 = [var_0, var_0, var_0, var_0, var_0, var_0]
    list_1 = [list_0, var_0, list_0, var_0]
    function_def_0 = module_0.FunctionDef(*list_1)
    return_from_generator_transformer_0 = module_1.ReturnFromGeneratorTransformer(
        list_1
    )
    function_def_1 = return_from_generator_transformer_0.visit_FunctionDef(
        function_def_0
    )


def test_case_6():
    var_0 = module_0.FunctionDef()
    list_0 = [var_0, var_0, var_0, var_0, var_0, var_0]
    list_1 = [list_0, var_0, list_0, var_0]
    var_1 = module_2.iter_fields(var_0)
    function_def_0 = module_0.FunctionDef(*list_1)
    return_from_generator_transformer_0 = module_1.ReturnFromGeneratorTransformer(
        function_def_0
    )
    return_from_generator_transformer_0.visit_FunctionDef(function_def_0)
