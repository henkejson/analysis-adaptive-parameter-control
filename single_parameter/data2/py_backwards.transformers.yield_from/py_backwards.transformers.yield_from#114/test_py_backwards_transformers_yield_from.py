# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.yield_from as module_1
import typed_ast.ast3 as module_2


def test_case_0():
    cmpop_0 = module_0.cmpop()
    yield_from_transformer_0 = module_1.YieldFromTransformer(cmpop_0)
    a_s_t_0 = yield_from_transformer_0.visit(cmpop_0)


def test_case_1():
    cmpop_0 = module_0.cmpop()
    yield_from_transformer_0 = module_1.YieldFromTransformer(cmpop_0)


def test_case_2():
    none_type_0 = None
    yield_from_transformer_0 = module_1.YieldFromTransformer(none_type_0)
    list_0 = [none_type_0, none_type_0]
    async_with_0 = module_0.AsyncWith(*list_0)
    var_0 = yield_from_transformer_0.visit(async_with_0)


def test_case_3():
    none_type_0 = None
    yield_from_transformer_0 = module_1.YieldFromTransformer(none_type_0)
    list_0 = [none_type_0, none_type_0]
    list_1 = [yield_from_transformer_0, list_0]
    async_with_0 = module_0.AsyncWith(*list_1)
    a_s_t_0 = yield_from_transformer_0.visit(async_with_0)
    module_2.increment_lineno(none_type_0)
