# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.yield_from as module_1


def test_case_0():
    index_0 = module_0.Index()
    yield_from_transformer_0 = module_1.YieldFromTransformer(index_0)
    yield_from_transformer_0.visit(yield_from_transformer_0)


def test_case_1():
    index_0 = module_0.Index()
    yield_from_transformer_0 = module_1.YieldFromTransformer(index_0)


def test_case_2():
    none_type_0 = None
    list_0 = [none_type_0, none_type_0]
    if_exp_0 = module_0.IfExp(*list_0)
    yield_from_transformer_0 = module_1.YieldFromTransformer(if_exp_0)
    a_s_t_0 = yield_from_transformer_0.visit(if_exp_0)


def test_case_3():
    int_0 = -4060
    yield_from_transformer_0 = module_1.YieldFromTransformer(int_0)
    list_0 = [int_0, int_0]
    list_1 = [yield_from_transformer_0, list_0, int_0]
    if_exp_0 = module_0.IfExp(*list_1)
    var_0 = yield_from_transformer_0.visit(if_exp_0)
