# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.dict_unpacking as module_1
import typed_ast.ast3 as module_2


def test_case_0():
    return_0 = module_0.Return()
    dict_unpacking_transformer_0 = module_1.DictUnpackingTransformer(return_0)


def test_case_1():
    str_0 = "\n5ZO#\r"
    list_0 = [str_0, str_0]
    dict_0 = module_0.Dict(*list_0)
    var_0 = module_2.iter_fields(str_0)
    dict_unpacking_transformer_0 = module_1.DictUnpackingTransformer(var_0)
    var_1 = module_2.walk(str_0)
    dict_unpacking_transformer_1 = module_1.DictUnpackingTransformer(var_1)
    dict_unpacking_transformer_2 = module_1.DictUnpackingTransformer(var_0)
    dict_unpacking_transformer_3 = module_1.DictUnpackingTransformer(var_1)
    dict_unpacking_transformer_0.visit_Module(var_1)
