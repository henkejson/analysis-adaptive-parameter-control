# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.dict_unpacking as module_1


def test_case_0():
    bit_and_0 = module_0.BitAnd()
    dict_unpacking_transformer_0 = module_1.DictUnpackingTransformer(bit_and_0)


def test_case_1():
    store_0 = module_0.Store()
    dict_unpacking_transformer_0 = module_1.DictUnpackingTransformer(store_0)
    dict_unpacking_transformer_1 = module_1.DictUnpackingTransformer(store_0)
    dict_unpacking_transformer_0.visit_Module(store_0)
