# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.dict_unpacking as module_1


def test_case_0():
    compare_0 = module_0.Compare()
    dict_unpacking_transformer_0 = module_1.DictUnpackingTransformer(compare_0)


def test_case_1():
    none_type_0 = None
    dict_unpacking_transformer_0 = module_1.DictUnpackingTransformer(none_type_0)
    dict_unpacking_transformer_0.visit_Module(none_type_0)
