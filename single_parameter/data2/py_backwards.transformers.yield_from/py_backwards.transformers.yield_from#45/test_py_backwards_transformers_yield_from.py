# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.yield_from as module_1
import typed_ast.ast3 as module_2


def test_case_0():
    unaryop_0 = module_0.unaryop()
    yield_from_transformer_0 = module_1.YieldFromTransformer(unaryop_0)
    a_s_t_0 = yield_from_transformer_0.visit(unaryop_0)


def test_case_1():
    unaryop_0 = module_0.unaryop()
    yield_from_transformer_0 = module_1.YieldFromTransformer(unaryop_0)


def test_case_2():
    none_type_0 = None
    list_0 = [none_type_0, none_type_0]
    with_0 = module_0.With(*list_0)
    yield_from_transformer_0 = module_1.YieldFromTransformer(none_type_0)
    a_s_t_0 = yield_from_transformer_0.visit(with_0)


def test_case_3():
    str_0 = "tkinter_dnd"
    none_type_0 = None
    yield_from_transformer_0 = module_1.YieldFromTransformer(none_type_0)
    var_0 = module_2.parse(str_0)
    a_s_t_0 = yield_from_transformer_0.visit(var_0)
