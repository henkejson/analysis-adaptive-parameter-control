# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.dict_unpacking as module_1


def test_case_0():
    async_for_0 = module_0.AsyncFor()
    dict_unpacking_transformer_0 = module_1.DictUnpackingTransformer(async_for_0)


def test_case_1():
    str_0 = "w"
    dict_unpacking_transformer_0 = module_1.DictUnpackingTransformer(str_0)
    none_type_0 = None
    dict_unpacking_transformer_0.visit_Module(none_type_0)
