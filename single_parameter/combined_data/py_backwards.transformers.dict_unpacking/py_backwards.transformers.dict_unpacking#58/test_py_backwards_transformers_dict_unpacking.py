# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.dict_unpacking as module_1
import typed_ast.ast3 as module_2


def test_case_0():
    dict_0 = module_0.Dict()
    dict_unpacking_transformer_0 = module_1.DictUnpackingTransformer(dict_0)


def test_case_1():
    type_ignore_0 = module_0.type_ignore()
    dict_unpacking_transformer_0 = module_1.DictUnpackingTransformer(type_ignore_0)
    var_0 = module_2.dump(type_ignore_0, dict_unpacking_transformer_0)
    var_1 = module_2.parse(var_0)
    module_0 = dict_unpacking_transformer_0.visit_Module(var_1)
