# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import py_backwards.transformers.dict_unpacking as module_0
import typed_ast._ast3 as module_1


def test_case_0():
    none_type_0 = None
    dict_unpacking_transformer_0 = module_0.DictUnpackingTransformer(none_type_0)


def test_case_1():
    a_s_t_0 = module_1.AST()
    dict_unpacking_transformer_0 = module_0.DictUnpackingTransformer(a_s_t_0)
    dict_unpacking_transformer_0.visit_Module(a_s_t_0)
