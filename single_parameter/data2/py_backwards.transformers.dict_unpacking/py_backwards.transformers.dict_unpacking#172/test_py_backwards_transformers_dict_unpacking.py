# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import py_backwards.transformers.dict_unpacking as module_0
import typed_ast._ast3 as module_1
import typed_ast.ast3 as module_2


def test_case_0():
    none_type_0 = None
    dict_unpacking_transformer_0 = module_0.DictUnpackingTransformer(none_type_0)


def test_case_1():
    nonlocal_0 = module_1.Nonlocal()
    dict_unpacking_transformer_0 = module_0.DictUnpackingTransformer(nonlocal_0)
    dict_unpacking_transformer_0.visit_Module(dict_unpacking_transformer_0)


def test_case_2():
    str_0 = "_4qE;"
    var_0 = module_2.parse(str_0)
    dict_unpacking_transformer_0 = module_0.DictUnpackingTransformer(str_0)
    module_0 = dict_unpacking_transformer_0.visit_Module(var_0)
