# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import py_backwards.transformers.yield_from as module_0
import typed_ast._ast3 as module_1
import typed_ast.ast3 as module_2


def test_case_0():
    none_type_0 = None
    yield_from_transformer_0 = module_0.YieldFromTransformer(none_type_0)
    yield_from_transformer_0.visit(yield_from_transformer_0)


def test_case_1():
    int_0 = -1261
    yield_from_transformer_0 = module_0.YieldFromTransformer(int_0)


def test_case_2():
    int_0 = -1298
    yield_from_transformer_0 = module_0.YieldFromTransformer(int_0)
    list_0 = [int_0, int_0, int_0]
    async_with_0 = module_1.AsyncWith(*list_0)
    a_s_t_0 = yield_from_transformer_0.visit(async_with_0)


def test_case_3():
    int_0 = -1335
    yield_from_transformer_0 = module_0.YieldFromTransformer(int_0)
    list_0 = [int_0, int_0, int_0]
    list_1 = [int_0, list_0]
    async_with_0 = module_1.AsyncWith(*list_1)
    a_s_t_0 = yield_from_transformer_0.visit(async_with_0)


def test_case_4():
    r_shift_0 = module_1.RShift()
    var_0 = module_2.dump(r_shift_0)
    var_1 = module_2.parse(var_0, var_0)
    yield_from_transformer_0 = module_0.YieldFromTransformer(var_1)
    a_s_t_0 = yield_from_transformer_0.visit(var_1)
    yield_from_transformer_0.visit(var_0)
