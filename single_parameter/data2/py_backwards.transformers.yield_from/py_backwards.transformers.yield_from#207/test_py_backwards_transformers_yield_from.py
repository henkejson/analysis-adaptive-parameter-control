# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.yield_from as module_1


def test_case_0():
    name_constant_0 = module_0.NameConstant()
    yield_from_transformer_0 = module_1.YieldFromTransformer(name_constant_0)
    a_s_t_0 = yield_from_transformer_0.visit(name_constant_0)


def test_case_1():
    ann_assign_0 = module_0.AnnAssign()
    yield_from_transformer_0 = module_1.YieldFromTransformer(ann_assign_0)


def test_case_2():
    str_0 = '"<\n+[\x0b'
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0}
    list_0 = [dict_0, dict_0]
    yield_from_transformer_0 = module_1.YieldFromTransformer(dict_0)
    module_0 = module_0.Module(*list_0)
    a_s_t_0 = yield_from_transformer_0.visit(module_0)
