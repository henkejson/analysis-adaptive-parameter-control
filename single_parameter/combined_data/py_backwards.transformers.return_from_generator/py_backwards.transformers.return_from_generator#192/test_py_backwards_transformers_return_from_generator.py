# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.return_from_generator as module_1
import typed_ast.ast3 as module_2


def test_case_0():
    await_0 = module_0.Await()
    await_0.visit_FunctionDef(await_0)


def test_case_1():
    str_0 = 'Skip transformer "{}"'
    none_type_0 = None
    return_from_generator_transformer_0 = module_1.ReturnFromGeneratorTransformer(
        none_type_0
    )
    list_0 = [return_from_generator_transformer_0, str_0, str_0, none_type_0]
    function_def_0 = module_0.FunctionDef(*list_0)
    function_def_1 = return_from_generator_transformer_0.visit_FunctionDef(
        function_def_0
    )


def test_case_2():
    str_0 = ""
    none_type_0 = None
    return_from_generator_transformer_0 = module_1.ReturnFromGeneratorTransformer(
        none_type_0
    )
    list_0 = [return_from_generator_transformer_0, str_0, str_0, none_type_0]
    function_def_0 = module_0.FunctionDef(*list_0)
    function_def_1 = return_from_generator_transformer_0.visit_FunctionDef(
        function_def_0
    )
    function_def_0.generic_visit(str_0)


def test_case_3():
    str_0 = "kN"
    none_type_0 = None
    return_from_generator_transformer_0 = module_1.ReturnFromGeneratorTransformer(
        none_type_0
    )
    function_def_0 = module_2.parse(str_0)
    function_def_1 = return_from_generator_transformer_0.visit_FunctionDef(
        function_def_0
    )
    function_def_0.generic_visit(str_0)
