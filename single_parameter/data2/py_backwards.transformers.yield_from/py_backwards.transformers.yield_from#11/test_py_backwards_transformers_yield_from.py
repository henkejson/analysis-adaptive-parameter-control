# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.yield_from as module_1


def test_case_0():
    slice_0 = module_0.slice()
    yield_from_transformer_0 = module_1.YieldFromTransformer(slice_0)
    a_s_t_0 = yield_from_transformer_0.visit(slice_0)


def test_case_1():
    index_0 = module_0.Index()
    yield_from_transformer_0 = module_1.YieldFromTransformer(index_0)


def test_case_2():
    none_type_0 = None
    yield_from_transformer_0 = module_1.YieldFromTransformer(none_type_0)
    list_0 = [none_type_0, yield_from_transformer_0, yield_from_transformer_0]
    str_0 = ",U=J#^H5"
    dict_0 = {
        str_0: none_type_0,
        str_0: none_type_0,
        str_0: none_type_0,
        str_0: none_type_0,
    }
    with_0 = module_0.With(*list_0, **dict_0)
    a_s_t_0 = yield_from_transformer_0.visit(with_0)
