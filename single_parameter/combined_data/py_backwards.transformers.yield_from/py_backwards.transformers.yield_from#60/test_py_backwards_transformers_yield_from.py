# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import py_backwards.transformers.yield_from as module_0
import typed_ast._ast3 as module_1
import typed_ast.ast3 as module_2


def test_case_0():
    bool_0 = False
    yield_from_transformer_0 = module_0.YieldFromTransformer(bool_0)
    yield_from_transformer_0.visit(bool_0)


def test_case_1():
    yield_0 = module_1.Yield()
    yield_from_transformer_0 = module_0.YieldFromTransformer(yield_0)


def test_case_2():
    bool_0 = True
    list_0 = [bool_0]
    interactive_0 = module_1.Interactive(*list_0)
    yield_from_transformer_0 = module_0.YieldFromTransformer(interactive_0)
    a_s_t_0 = yield_from_transformer_0.visit(interactive_0)


def test_case_3():
    bool_0 = True
    yield_from_transformer_0 = module_0.YieldFromTransformer(bool_0)
    list_0 = [bool_0]
    list_1 = [list_0]
    interactive_0 = module_1.Interactive(*list_1)
    a_s_t_0 = yield_from_transformer_0.visit(interactive_0)


def test_case_4():
    dict_0 = {}
    eq_0 = module_1.Eq(**dict_0)
    yield_from_transformer_0 = module_0.YieldFromTransformer(eq_0)
    bytes_0 = b"r"
    var_0 = module_2.parse(bytes_0)
    var_1 = yield_from_transformer_0.visit(var_0)
