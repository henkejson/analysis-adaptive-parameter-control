# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.yield_from as module_1
import typed_ast.ast3 as module_2


def test_case_0():
    invert_0 = module_0.Invert()
    yield_from_transformer_0 = module_1.YieldFromTransformer(invert_0)
    a_s_t_0 = yield_from_transformer_0.visit(invert_0)


def test_case_1():
    lt_e_0 = module_0.LtE()
    yield_from_transformer_0 = module_1.YieldFromTransformer(lt_e_0)


def test_case_2():
    alias_0 = module_0.alias()
    var_0 = module_2.dump(alias_0, alias_0)
    var_1 = module_2.parse(var_0, var_0)
    yield_from_transformer_0 = module_1.YieldFromTransformer(var_0)
    a_s_t_0 = yield_from_transformer_0.visit(var_1)
