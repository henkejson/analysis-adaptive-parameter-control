# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.yield_from as module_1


def test_case_0():
    slice_0 = module_0.slice()
    yield_from_transformer_0 = module_1.YieldFromTransformer(slice_0)
    a_s_t_0 = yield_from_transformer_0.visit(slice_0)


def test_case_1():
    slice_0 = module_0.slice()
    yield_from_transformer_0 = module_1.YieldFromTransformer(slice_0)


def test_case_2():
    type_ignore_0 = module_0.TypeIgnore()
    yield_from_transformer_0 = module_1.YieldFromTransformer(type_ignore_0)
    list_0 = [type_ignore_0, yield_from_transformer_0]
    with_0 = module_0.With(*list_0)
    a_s_t_0 = yield_from_transformer_0.visit(with_0)
