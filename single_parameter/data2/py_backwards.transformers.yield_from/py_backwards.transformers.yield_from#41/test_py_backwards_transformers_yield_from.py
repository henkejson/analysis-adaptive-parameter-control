# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import py_backwards.transformers.yield_from as module_0
import typed_ast._ast3 as module_1


def test_case_0():
    none_type_0 = None
    yield_from_transformer_0 = module_0.YieldFromTransformer(none_type_0)
    yield_from_transformer_0.visit(none_type_0)


def test_case_1():
    while_0 = module_1.While()
    yield_from_transformer_0 = module_0.YieldFromTransformer(while_0)


def test_case_2():
    dict_0 = {}
    list_0 = [dict_0]
    yield_from_transformer_0 = module_0.YieldFromTransformer(list_0)
    suite_0 = module_1.Suite(*list_0, **dict_0)
    var_0 = yield_from_transformer_0.visit(suite_0)
