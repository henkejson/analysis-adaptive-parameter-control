# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.dict_unpacking as module_1


def test_case_0():
    boolop_0 = module_0.boolop()
    dict_unpacking_transformer_0 = module_1.DictUnpackingTransformer(boolop_0)


def test_case_1():
    pow_0 = module_0.Pow()
    dict_unpacking_transformer_0 = module_1.DictUnpackingTransformer(pow_0)
    var_0 = dict_unpacking_transformer_0.visit(pow_0)
    dict_unpacking_transformer_1 = module_1.DictUnpackingTransformer(pow_0)
    dict_unpacking_transformer_1.visit_Module(dict_unpacking_transformer_0)
