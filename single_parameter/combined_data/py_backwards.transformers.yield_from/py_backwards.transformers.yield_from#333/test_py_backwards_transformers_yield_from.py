# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.yield_from as module_1
import typed_ast.ast3 as module_2


def test_case_0():
    bit_xor_0 = module_0.BitXor()
    yield_from_transformer_0 = module_1.YieldFromTransformer(bit_xor_0)
    a_s_t_0 = yield_from_transformer_0.visit(bit_xor_0)


def test_case_1():
    none_type_0 = None
    yield_from_transformer_0 = module_1.YieldFromTransformer(none_type_0)


def test_case_2():
    str_0 = "getstatusoutput"
    yield_from_transformer_0 = module_1.YieldFromTransformer(str_0)
    var_0 = module_2.parse(str_0)
    a_s_t_0 = yield_from_transformer_0.visit(var_0)


def test_case_3():
    str_0 = "Zo"
    dict_0 = {str_0: str_0, str_0: str_0}
    none_type_0 = None
    yield_from_transformer_0 = module_1.YieldFromTransformer(none_type_0)
    try_0 = module_0.Try(*dict_0)
    a_s_t_0 = yield_from_transformer_0.visit(try_0)
    module_2.get_docstring(yield_from_transformer_0)
