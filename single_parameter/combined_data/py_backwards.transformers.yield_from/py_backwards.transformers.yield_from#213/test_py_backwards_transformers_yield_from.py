# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.yield_from as module_1


def test_case_0():
    suite_0 = module_0.Suite()
    yield_from_transformer_0 = module_1.YieldFromTransformer(suite_0)
    var_0 = yield_from_transformer_0.visit(suite_0)


def test_case_1():
    bool_0 = False
    yield_from_transformer_0 = module_1.YieldFromTransformer(bool_0)


def test_case_2():
    none_type_0 = None
    yield_from_transformer_0 = module_1.YieldFromTransformer(none_type_0)
    list_0 = [yield_from_transformer_0, yield_from_transformer_0]
    module_0 = module_0.Module(*list_0)
    a_s_t_0 = yield_from_transformer_0.visit(module_0)


def test_case_3():
    none_type_0 = None
    yield_from_transformer_0 = module_1.YieldFromTransformer(none_type_0)
    list_0 = [yield_from_transformer_0, yield_from_transformer_0]
    list_1 = [list_0]
    module_0 = module_0.Module(*list_1)
    a_s_t_0 = yield_from_transformer_0.visit(module_0)
