# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.dict_unpacking as module_1
import typed_ast.ast3 as module_2


def test_case_0():
    pass_0 = module_0.Pass()
    dict_unpacking_transformer_0 = module_1.DictUnpackingTransformer(pass_0)


def test_case_1():
    load_0 = module_0.Load()
    dict_unpacking_transformer_0 = module_1.DictUnpackingTransformer(load_0)
    dict_unpacking_transformer_0.visit_Module(dict_unpacking_transformer_0)


def test_case_2():
    str_0 = "ij0"
    var_0 = module_2.parse(str_0)
    dict_unpacking_transformer_0 = module_1.DictUnpackingTransformer(var_0)
    module_0 = dict_unpacking_transformer_0.visit_Module(var_0)
