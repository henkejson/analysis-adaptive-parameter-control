# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.yield_from as module_1


def test_case_0():
    expr_context_0 = module_0.expr_context()
    yield_from_transformer_0 = module_1.YieldFromTransformer(expr_context_0)
    a_s_t_0 = yield_from_transformer_0.visit(expr_context_0)


def test_case_1():
    expr_context_0 = module_0.expr_context()
    yield_from_transformer_0 = module_1.YieldFromTransformer(expr_context_0)


def test_case_2():
    aug_assign_0 = module_0.AugAssign()
    list_0 = [aug_assign_0, aug_assign_0]
    if_0 = module_0.If(*list_0)
    yield_from_transformer_0 = module_1.YieldFromTransformer(aug_assign_0)
    a_s_t_0 = yield_from_transformer_0.visit(if_0)
