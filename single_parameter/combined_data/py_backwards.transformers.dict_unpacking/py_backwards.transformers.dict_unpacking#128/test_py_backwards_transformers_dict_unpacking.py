# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import py_backwards.transformers.dict_unpacking as module_0
import typed_ast._ast3 as module_1
import typed_ast.ast3 as module_2


def test_case_0():
    none_type_0 = None
    dict_unpacking_transformer_0 = module_0.DictUnpackingTransformer(none_type_0)


def test_case_1():
    subscript_0 = module_1.Subscript()
    dict_unpacking_transformer_0 = module_0.DictUnpackingTransformer(subscript_0)
    raise_0 = module_1.Raise()
    dict_unpacking_transformer_1 = module_0.DictUnpackingTransformer(raise_0)
    var_0 = module_2.walk(dict_unpacking_transformer_1)
    dict_unpacking_transformer_2 = module_0.DictUnpackingTransformer(var_0)
    dict_unpacking_transformer_3 = module_0.DictUnpackingTransformer(raise_0)
    dict_unpacking_transformer_4 = module_0.DictUnpackingTransformer(
        dict_unpacking_transformer_1
    )
    dict_unpacking_transformer_2.visit_Module(raise_0)
