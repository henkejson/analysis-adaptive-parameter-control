# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.yield_from as module_1


def test_case_0():
    bit_xor_0 = module_0.BitXor()
    yield_from_transformer_0 = module_1.YieldFromTransformer(bit_xor_0)
    a_s_t_0 = yield_from_transformer_0.visit(bit_xor_0)


def test_case_1():
    none_type_0 = None
    yield_from_transformer_0 = module_1.YieldFromTransformer(none_type_0)


def test_case_2():
    eq_0 = module_0.Eq()
    yield_from_transformer_0 = module_1.YieldFromTransformer(eq_0)
    list_0 = [yield_from_transformer_0, eq_0, yield_from_transformer_0, eq_0]
    function_def_0 = module_0.FunctionDef(*list_0)
    a_s_t_0 = yield_from_transformer_0.visit(function_def_0)
