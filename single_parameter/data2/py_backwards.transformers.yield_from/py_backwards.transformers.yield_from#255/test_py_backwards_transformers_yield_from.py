# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.yield_from as module_1
import typed_ast.ast3 as module_2


def test_case_0():
    is_not_0 = module_0.IsNot()
    yield_from_transformer_0 = module_1.YieldFromTransformer(is_not_0)
    a_s_t_0 = yield_from_transformer_0.visit(is_not_0)


def test_case_1():
    is_not_0 = module_0.IsNot()
    yield_from_transformer_0 = module_1.YieldFromTransformer(is_not_0)


def test_case_2():
    bool_0 = True
    list_0 = [bool_0, bool_0, bool_0, bool_0]
    async_function_def_0 = module_0.AsyncFunctionDef(*list_0)
    yield_from_transformer_0 = module_1.YieldFromTransformer(async_function_def_0)
    a_s_t_0 = yield_from_transformer_0.visit(async_function_def_0)


def test_case_3():
    bool_0 = True
    yield_from_transformer_0 = module_1.YieldFromTransformer(bool_0)
    try_0 = module_0.Try()
    var_0 = module_2.dump(try_0)
    var_1 = module_2.parse(var_0)
    a_s_t_0 = yield_from_transformer_0.visit(var_1)
