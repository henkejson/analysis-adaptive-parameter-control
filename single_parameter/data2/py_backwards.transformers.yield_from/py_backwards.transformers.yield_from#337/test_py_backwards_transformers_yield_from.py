# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.yield_from as module_1
import typed_ast.ast3 as module_2


def test_case_0():
    suite_0 = module_0.Suite()
    yield_from_transformer_0 = module_1.YieldFromTransformer(suite_0)
    a_s_t_0 = yield_from_transformer_0.visit(suite_0)


def test_case_1():
    not_in_0 = module_0.NotIn()
    yield_from_transformer_0 = module_1.YieldFromTransformer(not_in_0)


def test_case_2():
    str_0 = "Tkdn"
    yield_from_transformer_0 = module_1.YieldFromTransformer(str_0)
    var_0 = module_2.parse(str_0)
    a_s_t_0 = yield_from_transformer_0.visit(var_0)


def test_case_3():
    async_for_0 = module_0.AsyncFor()
    yield_from_transformer_0 = module_1.YieldFromTransformer(async_for_0)
    list_0 = [async_for_0, async_for_0, async_for_0]
    async_function_def_0 = module_0.AsyncFunctionDef(*list_0)
    a_s_t_0 = yield_from_transformer_0.visit(async_function_def_0)
