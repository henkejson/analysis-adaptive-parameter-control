# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.yield_from as module_1
import typed_ast.ast3 as module_2


def test_case_0():
    compare_0 = module_0.Compare()
    yield_from_transformer_0 = module_1.YieldFromTransformer(compare_0)
    a_s_t_0 = yield_from_transformer_0.visit(compare_0)


def test_case_1():
    compare_0 = module_0.Compare()
    yield_from_transformer_0 = module_1.YieldFromTransformer(compare_0)


def test_case_2():
    str_0 = "tkinter0_clorchooser"
    var_0 = module_2.parse(str_0)
    yield_from_transformer_0 = module_1.YieldFromTransformer(var_0)
    a_s_t_0 = yield_from_transformer_0.visit(var_0)


def test_case_3():
    a_s_t_0 = module_0.AST()
    list_0 = [a_s_t_0, a_s_t_0, a_s_t_0, a_s_t_0, a_s_t_0]
    async_function_def_0 = module_0.AsyncFunctionDef(*list_0)
    yield_from_transformer_0 = module_1.YieldFromTransformer(list_0)
    a_s_t_1 = yield_from_transformer_0.visit(async_function_def_0)
