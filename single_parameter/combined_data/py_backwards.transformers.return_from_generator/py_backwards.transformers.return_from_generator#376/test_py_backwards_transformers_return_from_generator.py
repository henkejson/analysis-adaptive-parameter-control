# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import typed_ast.ast3 as module_1
import py_backwards.transformers.return_from_generator as module_2


def test_case_0():
    set_0 = module_0.Set()
    module_1.get_docstring(set_0)


def test_case_1():
    bytes_0 = b"Gf"
    var_0 = module_1.parse(bytes_0)
    return_from_generator_transformer_0 = module_2.ReturnFromGeneratorTransformer(var_0)
    function_def_0 = return_from_generator_transformer_0.visit_FunctionDef(var_0)


def test_case_2():
    bytes_0 = b""
    var_0 = module_1.parse(bytes_0)
    return_from_generator_transformer_0 = module_2.ReturnFromGeneratorTransformer(var_0)
    function_def_0 = return_from_generator_transformer_0.visit_FunctionDef(var_0)
