# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.return_from_generator as module_1


def test_case_0():
    floor_div_0 = module_0.FloorDiv()
    return_from_generator_transformer_0 = module_1.ReturnFromGeneratorTransformer(
        floor_div_0
    )


def test_case_1():
    str_0 = "-!"
    return_from_generator_transformer_0 = module_1.ReturnFromGeneratorTransformer(str_0)
    list_0 = [str_0, str_0, str_0]
    function_def_0 = module_0.FunctionDef(*list_0)
    var_0 = return_from_generator_transformer_0.visit(function_def_0)


def test_case_2():
    str_0 = ""
    none_type_0 = None
    return_from_generator_transformer_0 = module_1.ReturnFromGeneratorTransformer(
        none_type_0
    )
    list_0 = [none_type_0, none_type_0, str_0, str_0, none_type_0]
    function_def_0 = module_0.FunctionDef(*list_0)
    function_def_1 = return_from_generator_transformer_0.visit_FunctionDef(
        function_def_0
    )
