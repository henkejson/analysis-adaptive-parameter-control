# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.dict_unpacking as module_1


def test_case_0():
    arg_0 = module_0.arg()
    dict_unpacking_transformer_0 = module_1.DictUnpackingTransformer(arg_0)


def test_case_1():
    aug_store_0 = module_0.AugStore()
    dict_unpacking_transformer_0 = module_1.DictUnpackingTransformer(aug_store_0)
    dict_unpacking_transformer_0.visit_Module(dict_unpacking_transformer_0)
