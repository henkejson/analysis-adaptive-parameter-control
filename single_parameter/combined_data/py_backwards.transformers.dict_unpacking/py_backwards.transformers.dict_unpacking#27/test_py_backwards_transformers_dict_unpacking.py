# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import py_backwards.transformers.dict_unpacking as module_0
import typed_ast.ast3 as module_1


def test_case_0():
    bool_0 = False
    dict_unpacking_transformer_0 = module_0.DictUnpackingTransformer(bool_0)


def test_case_1():
    str_0 = "\x0cXi"
    var_0 = module_1.parse(str_0)
    dict_unpacking_transformer_0 = module_0.DictUnpackingTransformer(str_0)
    module_0 = dict_unpacking_transformer_0.visit_Module(var_0)
