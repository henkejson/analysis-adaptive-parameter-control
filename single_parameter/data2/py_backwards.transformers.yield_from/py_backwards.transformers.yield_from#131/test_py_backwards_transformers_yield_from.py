# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import py_backwards.transformers.yield_from as module_0
import typed_ast._ast3 as module_1


def test_case_0():
    none_type_0 = None
    yield_from_transformer_0 = module_0.YieldFromTransformer(none_type_0)
    yield_from_transformer_0.visit(yield_from_transformer_0)


def test_case_1():
    suite_0 = module_1.Suite()
    yield_from_transformer_0 = module_0.YieldFromTransformer(suite_0)


def test_case_2():
    bytes_0 = b"\x9ef\xa7"
    yield_from_transformer_0 = module_0.YieldFromTransformer(bytes_0)
    async_with_0 = module_1.AsyncWith(*bytes_0)
    a_s_t_0 = yield_from_transformer_0.visit(async_with_0)
