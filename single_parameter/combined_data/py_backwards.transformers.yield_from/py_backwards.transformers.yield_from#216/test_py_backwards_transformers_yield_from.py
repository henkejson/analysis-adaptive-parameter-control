# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import py_backwards.transformers.yield_from as module_0
import typed_ast.ast3 as module_1
import typed_ast._ast3 as module_2


def test_case_0():
    none_type_0 = None
    yield_from_transformer_0 = module_0.YieldFromTransformer(none_type_0)
    yield_from_transformer_0.visit(none_type_0)


def test_case_1():
    none_type_0 = None
    module_1.dump(none_type_0)


def test_case_2():
    str_0 = "ProxyHandler"
    yield_from_transformer_0 = module_0.YieldFromTransformer(str_0)
    var_0 = module_1.parse(str_0)
    a_s_t_0 = yield_from_transformer_0.visit(var_0)


def test_case_3():
    str_0 = "ProxyHandler"
    yield_from_transformer_0 = module_0.YieldFromTransformer(str_0)
    list_0 = [str_0, str_0, str_0]
    yield_from_transformer_1 = module_0.YieldFromTransformer(str_0)
    if_exp_0 = module_2.IfExp(*list_0)
    a_s_t_0 = yield_from_transformer_0.visit(if_exp_0)
