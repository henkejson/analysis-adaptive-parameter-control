# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.yield_from as module_1
import typed_ast.ast3 as module_2


def test_case_0():
    load_0 = module_0.Load()
    yield_from_transformer_0 = module_1.YieldFromTransformer(load_0)
    yield_from_transformer_0.visit(yield_from_transformer_0)


def test_case_1():
    load_0 = module_0.Load()
    yield_from_transformer_0 = module_1.YieldFromTransformer(load_0)


def test_case_2():
    str_0 = "email_mime_text"
    var_0 = module_2.parse(str_0)
    yield_from_transformer_0 = module_1.YieldFromTransformer(var_0)
    a_s_t_0 = yield_from_transformer_0.visit(var_0)


def test_case_3():
    unary_op_0 = module_0.UnaryOp()
    var_0 = module_2.walk(unary_op_0)
    yield_from_transformer_0 = module_1.YieldFromTransformer(var_0)
    suite_0 = module_0.Suite(*var_0)
    a_s_t_0 = yield_from_transformer_0.visit(suite_0)
