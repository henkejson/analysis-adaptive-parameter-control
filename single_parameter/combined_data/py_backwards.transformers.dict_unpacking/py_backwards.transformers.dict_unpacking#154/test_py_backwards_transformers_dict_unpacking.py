# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import py_backwards.transformers.dict_unpacking as module_0
import typed_ast._ast3 as module_1


def test_case_0():
    none_type_0 = None
    dict_unpacking_transformer_0 = module_0.DictUnpackingTransformer(none_type_0)


def test_case_1():
    load_0 = module_1.Load()
    dict_unpacking_transformer_0 = module_0.DictUnpackingTransformer(load_0)
    dict_unpacking_transformer_0.visit_Module(dict_unpacking_transformer_0)
