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
    none_type_0 = None
    yield_from_transformer_0 = module_0.YieldFromTransformer(none_type_0)


def test_case_2():
    none_type_0 = None
    yield_from_transformer_0 = module_0.YieldFromTransformer(none_type_0)
    list_0 = [
        none_type_0,
        yield_from_transformer_0,
        yield_from_transformer_0,
        yield_from_transformer_0,
    ]
    async_function_def_0 = module_1.AsyncFunctionDef(*list_0)
    a_s_t_0 = yield_from_transformer_0.visit(async_function_def_0)


def test_case_3():
    nonlocal_0 = module_1.Nonlocal()
    list_0 = [nonlocal_0]
    list_1 = [list_0]
    suite_0 = module_1.Suite(*list_1)
    yield_from_transformer_0 = module_0.YieldFromTransformer(suite_0)
    a_s_t_0 = yield_from_transformer_0.visit(suite_0)


def test_case_4():
    async_for_0 = module_1.AsyncFor()
    str_0 = "U2a0"
    yield_from_transformer_0 = module_0.YieldFromTransformer(async_for_0)
    var_0 = module_2.parse(str_0)
    a_s_t_0 = yield_from_transformer_0.visit(var_0)
