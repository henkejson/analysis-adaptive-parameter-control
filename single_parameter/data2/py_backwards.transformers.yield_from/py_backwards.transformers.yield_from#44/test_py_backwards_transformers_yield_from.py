# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.yield_from as module_1
import typed_ast.ast3 as module_2


def test_case_0():
    list_0 = module_0.List()
    yield_from_transformer_0 = module_1.YieldFromTransformer(list_0)
    a_s_t_0 = yield_from_transformer_0.visit(list_0)


def test_case_1():
    list_0 = module_0.List()
    yield_from_transformer_0 = module_1.YieldFromTransformer(list_0)


def test_case_2():
    str_0 = "tkSimpleDialog"
    var_0 = module_2.parse(str_0)
    yield_from_transformer_0 = module_1.YieldFromTransformer(var_0)
    var_1 = yield_from_transformer_0.visit(var_0)


def test_case_3():
    none_type_0 = None
    list_0 = [none_type_0]
    expression_0 = module_0.Expression(*list_0)
    yield_from_transformer_0 = module_1.YieldFromTransformer(none_type_0)
    a_s_t_0 = yield_from_transformer_0.visit(expression_0)
