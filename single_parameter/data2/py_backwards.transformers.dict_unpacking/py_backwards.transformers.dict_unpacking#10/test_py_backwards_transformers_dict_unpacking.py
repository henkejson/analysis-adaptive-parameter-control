# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.dict_unpacking as module_1
import typed_ast.ast3 as module_2


def test_case_0():
    keyword_0 = module_0.keyword()
    dict_unpacking_transformer_0 = module_1.DictUnpackingTransformer(keyword_0)


def test_case_1():
    str_0 = "yc;"
    var_0 = module_2.parse(str_0)
    dict_unpacking_transformer_0 = module_1.DictUnpackingTransformer(var_0)
    module_0 = dict_unpacking_transformer_0.visit_Module(var_0)
