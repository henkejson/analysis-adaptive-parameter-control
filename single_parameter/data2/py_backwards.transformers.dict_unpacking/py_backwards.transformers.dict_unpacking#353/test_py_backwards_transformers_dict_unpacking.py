# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import py_backwards.transformers.dict_unpacking as module_0
import typed_ast._ast3 as module_1


def test_case_0():
    none_type_0 = None
    dict_unpacking_transformer_0 = module_0.DictUnpackingTransformer(none_type_0)


def test_case_1():
    list_0 = []
    is_0 = module_1.Is(*list_0)
    dict_unpacking_transformer_0 = module_0.DictUnpackingTransformer(is_0)
    dict_unpacking_transformer_0.visit_Module(is_0)


def test_case_2():
    list_0 = []
    none_type_0 = None
    str_0 = "body"
    dict_0 = {str_0: none_type_0, str_0: list_0}
    module_0 = module_1.Module(**dict_0)
    dict_unpacking_transformer_0 = module_0.DictUnpackingTransformer(list_0)
    module_1 = dict_unpacking_transformer_0.visit_Module(module_0)
