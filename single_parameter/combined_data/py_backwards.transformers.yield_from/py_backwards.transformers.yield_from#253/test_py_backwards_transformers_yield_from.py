# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.yield_from as module_1


def test_case_0():
    withitem_0 = module_0.withitem()
    yield_from_transformer_0 = module_1.YieldFromTransformer(withitem_0)
    a_s_t_0 = yield_from_transformer_0.visit(withitem_0)


def test_case_1():
    is_0 = module_0.Is()
    yield_from_transformer_0 = module_1.YieldFromTransformer(is_0)


def test_case_2():
    bool_0 = True
    yield_from_transformer_0 = module_1.YieldFromTransformer(bool_0)
    list_0 = [bool_0]
    expression_0 = module_0.Expression(*list_0)
    a_s_t_0 = yield_from_transformer_0.visit(expression_0)


def test_case_3():
    bool_0 = True
    yield_from_transformer_0 = module_1.YieldFromTransformer(bool_0)
    list_0 = [bool_0]
    list_1 = [list_0]
    expression_0 = module_0.Expression(*list_1)
    a_s_t_0 = yield_from_transformer_0.visit(expression_0)
