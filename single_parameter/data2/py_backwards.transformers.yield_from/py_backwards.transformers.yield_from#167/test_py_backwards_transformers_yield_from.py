# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import py_backwards.transformers.yield_from as module_0
import typed_ast._ast3 as module_1


def test_case_0():
    none_type_0 = None
    yield_from_transformer_0 = module_0.YieldFromTransformer(none_type_0)
    yield_from_transformer_0.visit(none_type_0)


def test_case_1():
    mod_0 = module_1.mod()
    yield_from_transformer_0 = module_0.YieldFromTransformer(mod_0)


def test_case_2():
    starred_0 = module_1.Starred()
    yield_from_transformer_0 = module_0.YieldFromTransformer(starred_0)
    list_0 = [starred_0, yield_from_transformer_0, starred_0]
    function_def_0 = module_1.FunctionDef(*list_0)
    a_s_t_0 = yield_from_transformer_0.visit(function_def_0)
