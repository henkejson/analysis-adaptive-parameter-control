# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import py_backwards.transformers.yield_from as module_0
import typed_ast._ast3 as module_1
import typed_ast.ast3 as module_2


def test_case_0():
    int_0 = -2726
    yield_from_transformer_0 = module_0.YieldFromTransformer(int_0)
    yield_from_transformer_0.visit(yield_from_transformer_0)


def test_case_1():
    formatted_value_0 = module_1.FormattedValue()
    yield_from_transformer_0 = module_0.YieldFromTransformer(formatted_value_0)


def test_case_2():
    none_type_0 = None
    yield_from_transformer_0 = module_0.YieldFromTransformer(none_type_0)
    list_0 = [
        none_type_0,
        yield_from_transformer_0,
        yield_from_transformer_0,
        none_type_0,
    ]
    try_0 = module_1.Try(*list_0)
    a_s_t_0 = yield_from_transformer_0.visit(try_0)


def test_case_3():
    none_type_0 = None
    yield_from_transformer_0 = module_0.YieldFromTransformer(none_type_0)
    str_0 = "_py_backwards_merge_dicts"
    var_0 = module_2.parse(str_0)
    a_s_t_0 = yield_from_transformer_0.visit(var_0)
