# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import py_backwards.transformers.return_from_generator as module_0
import typed_ast._ast3 as module_1
import typed_ast.ast3 as module_2


def test_case_0():
    none_type_0 = None
    return_from_generator_transformer_0 = module_0.ReturnFromGeneratorTransformer(
        none_type_0
    )


def test_case_1():
    dict_0 = {}
    arg_0 = module_1.arg(**dict_0)
    var_0 = module_2.dump(arg_0, arg_0)
    return_from_generator_transformer_0 = module_0.ReturnFromGeneratorTransformer(var_0)
    var_1 = module_2.parse(var_0)
    function_def_0 = return_from_generator_transformer_0.visit_FunctionDef(var_1)


def test_case_2():
    str_0 = ""
    dict_0 = {str_0: str_0}
    arg_0 = module_1.arg(**dict_0)
    return_from_generator_transformer_0 = module_0.ReturnFromGeneratorTransformer(arg_0)
    var_0 = module_2.parse(str_0)
    function_def_0 = return_from_generator_transformer_0.visit_FunctionDef(var_0)
    var_1 = module_2.increment_lineno(var_0)
    return_from_generator_transformer_1 = module_0.ReturnFromGeneratorTransformer(var_1)
    function_def_1 = return_from_generator_transformer_0.visit_FunctionDef(var_0)
