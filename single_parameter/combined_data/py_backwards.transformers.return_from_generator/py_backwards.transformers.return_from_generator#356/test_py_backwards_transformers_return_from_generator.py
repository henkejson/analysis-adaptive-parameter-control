# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.return_from_generator as module_1
import typed_ast.ast3 as module_2


def test_case_0():
    not_eq_0 = module_0.NotEq()
    not_eq_0.visit_FunctionDef(not_eq_0)


def test_case_1():
    type_ignore_0 = module_0.TypeIgnore()
    return_from_generator_transformer_0 = module_1.ReturnFromGeneratorTransformer(
        type_ignore_0
    )
    var_0 = module_2.dump(
        type_ignore_0, type_ignore_0, return_from_generator_transformer_0
    )
    var_1 = module_2.parse(var_0)
    function_def_0 = return_from_generator_transformer_0.visit_FunctionDef(var_1)
