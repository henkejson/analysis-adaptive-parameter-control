# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.yield_from as module_1


def test_case_0():
    bit_xor_0 = module_0.BitXor()
    yield_from_transformer_0 = module_1.YieldFromTransformer(bit_xor_0)
    yield_from_transformer_0.visit(yield_from_transformer_0)


def test_case_1():
    bit_xor_0 = module_0.BitXor()
    yield_from_transformer_0 = module_1.YieldFromTransformer(bit_xor_0)


def test_case_2():
    none_type_0 = None
    list_0 = [none_type_0]
    expression_0 = module_0.Expression(*list_0)
    yield_from_transformer_0 = module_1.YieldFromTransformer(none_type_0)
    a_s_t_0 = yield_from_transformer_0.visit(expression_0)
