# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import py_backwards.transformers.dict_unpacking as module_0
import typed_ast._ast3 as module_1


def test_case_0():
    none_type_0 = None
    dict_unpacking_transformer_0 = module_0.DictUnpackingTransformer(none_type_0)


def test_case_1():
    str_0 = "B\x0cO*I"
    subscript_0 = module_1.Subscript()
    str_1 = ">&gw"
    dict_0 = {str_0: str_0, str_1: str_1, str_1: str_0, str_0: str_0}
    dict_unpacking_transformer_0 = module_0.DictUnpackingTransformer(dict_0)
    dict_unpacking_transformer_1 = module_0.DictUnpackingTransformer(str_1)
    list_0 = [dict_0]
    dict_1 = module_1.Dict(*list_0, **dict_0)
    var_0 = dict_unpacking_transformer_0.visit_Dict(dict_1)
    dict_unpacking_transformer_1.visit_Module(dict_0)


def test_case_2():
    str_0 = ">&gw"
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0, str_0: str_0}
    dict_unpacking_transformer_0 = module_0.DictUnpackingTransformer(dict_0)
    list_0 = [dict_0]
    dict_1 = module_1.Dict(*list_0, **dict_0)
    var_0 = dict_unpacking_transformer_0.visit_Dict(dict_1)
