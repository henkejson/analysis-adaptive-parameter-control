# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import py_backwards.transformers.dict_unpacking as module_0
import typed_ast.ast3 as module_1


def test_case_0():
    none_type_0 = None
    dict_unpacking_transformer_0 = module_0.DictUnpackingTransformer(none_type_0)


def test_case_1():
    bytes_0 = b"#OWK\xe2\xd3R\xebqN\xc8\xdd\xa5|\x04);\xc4TX"
    dict_unpacking_transformer_0 = module_0.DictUnpackingTransformer(bytes_0)
    var_0 = module_1.parse(bytes_0)
    module_0 = dict_unpacking_transformer_0.visit_Module(var_0)
