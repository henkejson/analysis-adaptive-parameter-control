# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.yield_from as module_1


def test_case_0():
    list_0 = module_0.List()
    yield_from_transformer_0 = module_1.YieldFromTransformer(list_0)
    yield_from_transformer_0.visit(yield_from_transformer_0)


def test_case_1():
    none_type_0 = None
    yield_from_transformer_0 = module_1.YieldFromTransformer(none_type_0)


def test_case_2():
    none_type_0 = None
    list_0 = [none_type_0, none_type_0, none_type_0]
    if_0 = module_0.If(*list_0)
    yield_from_transformer_0 = module_1.YieldFromTransformer(if_0)
    a_s_t_0 = yield_from_transformer_0.visit(if_0)


def test_case_3():
    none_type_0 = None
    list_0 = [none_type_0, none_type_0, none_type_0]
    yield_from_transformer_0 = module_1.YieldFromTransformer(none_type_0)
    list_1 = [none_type_0, none_type_0, list_0]
    async_for_0 = module_0.AsyncFor(*list_1)
    a_s_t_0 = yield_from_transformer_0.visit(async_for_0)
