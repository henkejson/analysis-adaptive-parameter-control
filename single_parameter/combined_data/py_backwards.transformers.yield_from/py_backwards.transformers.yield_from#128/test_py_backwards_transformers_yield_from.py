# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.yield_from as module_1
import typed_ast.ast3 as module_2


def test_case_0():
    keyword_0 = module_0.keyword()
    yield_from_transformer_0 = module_1.YieldFromTransformer(keyword_0)
    a_s_t_0 = yield_from_transformer_0.visit(keyword_0)


def test_case_1():
    none_type_0 = None
    module_2.dump(none_type_0)


def test_case_2():
    keyword_0 = module_0.keyword()
    yield_from_transformer_0 = module_1.YieldFromTransformer(keyword_0)
    list_0 = [yield_from_transformer_0]
    module_0 = module_0.Module(*list_0)
    a_s_t_0 = yield_from_transformer_0.visit(module_0)


def test_case_3():
    keyword_0 = module_0.keyword()
    yield_from_transformer_0 = module_1.YieldFromTransformer(keyword_0)
    list_0 = [yield_from_transformer_0, yield_from_transformer_0]
    list_1 = [list_0, keyword_0]
    try_0 = module_0.Try(*list_1)
    a_s_t_0 = yield_from_transformer_0.visit(try_0)
