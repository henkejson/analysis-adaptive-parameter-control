# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.yield_from as module_1


def test_case_0():
    ann_assign_0 = module_0.AnnAssign()
    yield_from_transformer_0 = module_1.YieldFromTransformer(ann_assign_0)
    a_s_t_0 = yield_from_transformer_0.visit(ann_assign_0)


def test_case_1():
    floor_div_0 = module_0.FloorDiv()
    yield_from_transformer_0 = module_1.YieldFromTransformer(floor_div_0)


def test_case_2():
    nonlocal_0 = module_0.Nonlocal()
    list_0 = [nonlocal_0]
    suite_0 = module_0.Suite(*list_0)
    yield_from_transformer_0 = module_1.YieldFromTransformer(nonlocal_0)
    a_s_t_0 = yield_from_transformer_0.visit(suite_0)
