# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.yield_from as module_1


def test_case_0():
    mult_0 = module_0.Mult()
    yield_from_transformer_0 = module_1.YieldFromTransformer(mult_0)
    a_s_t_0 = yield_from_transformer_0.visit(mult_0)


def test_case_1():
    mult_0 = module_0.Mult()
    yield_from_transformer_0 = module_1.YieldFromTransformer(mult_0)


def test_case_2():
    async_for_0 = module_0.AsyncFor()
    yield_from_transformer_0 = module_1.YieldFromTransformer(async_for_0)
    list_0 = [
        yield_from_transformer_0,
        yield_from_transformer_0,
        yield_from_transformer_0,
        async_for_0,
    ]
    for_0 = module_0.For(*list_0)
    a_s_t_0 = yield_from_transformer_0.visit(for_0)


def test_case_3():
    list_0 = []
    async_for_0 = module_0.AsyncFor()
    yield_from_transformer_0 = module_1.YieldFromTransformer(async_for_0)
    list_1 = [yield_from_transformer_0, async_for_0, list_0]
    for_0 = module_0.For(*list_1)
    a_s_t_0 = yield_from_transformer_0.visit(for_0)
