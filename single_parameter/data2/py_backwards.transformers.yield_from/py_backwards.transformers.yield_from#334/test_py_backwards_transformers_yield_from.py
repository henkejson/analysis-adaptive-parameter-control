# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.yield_from as module_1
import typed_ast.ast3 as module_2


def test_case_0():
    nonlocal_0 = module_0.Nonlocal()
    yield_from_transformer_0 = module_1.YieldFromTransformer(nonlocal_0)
    a_s_t_0 = yield_from_transformer_0.visit(nonlocal_0)


def test_case_1():
    nonlocal_0 = module_0.Nonlocal()
    yield_from_transformer_0 = module_1.YieldFromTransformer(nonlocal_0)


def test_case_2():
    str_0 = "time"
    var_0 = module_2.parse(str_0)
    yield_from_transformer_0 = module_1.YieldFromTransformer(var_0)
    a_s_t_0 = yield_from_transformer_0.visit(var_0)


def test_case_3():
    nonlocal_0 = module_0.Nonlocal()
    list_0 = [nonlocal_0]
    try_0 = module_0.Try(*list_0)
    yield_from_transformer_0 = module_1.YieldFromTransformer(nonlocal_0)
    a_s_t_0 = yield_from_transformer_0.visit(try_0)
