# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.dict_unpacking as module_1


def test_case_0():
    starred_0 = module_0.Starred()
    dict_unpacking_transformer_0 = module_1.DictUnpackingTransformer(starred_0)


def test_case_1():
    bytes_0 = b"2\x1c"
    dict_unpacking_transformer_0 = module_1.DictUnpackingTransformer(bytes_0)
    dict_unpacking_transformer_0.visit_Module(dict_unpacking_transformer_0)
