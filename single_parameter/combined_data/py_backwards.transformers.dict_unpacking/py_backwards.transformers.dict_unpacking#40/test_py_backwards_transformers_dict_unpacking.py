# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import py_backwards.transformers.dict_unpacking as module_0
import typed_ast._ast3 as module_1
import typed_ast.ast3 as module_2


def test_case_0():
    none_type_0 = None
    dict_unpacking_transformer_0 = module_0.DictUnpackingTransformer(none_type_0)


def test_case_1():
    joined_str_0 = module_1.JoinedStr()
    var_0 = module_2.walk(joined_str_0)
    list_0 = [var_0, var_0]
    dict_0 = module_1.Dict(*list_0)
    dict_unpacking_transformer_0 = module_0.DictUnpackingTransformer(joined_str_0)
    var_1 = dict_unpacking_transformer_0.visit_Dict(dict_0)
    dict_unpacking_transformer_0.visit_Module(var_0)


def test_case_2():
    joined_str_0 = module_1.JoinedStr()
    var_0 = module_2.iter_child_nodes(joined_str_0)
    list_0 = [var_0, var_0]
    dict_0 = module_1.Dict(*list_0)
    dict_unpacking_transformer_0 = module_0.DictUnpackingTransformer(joined_str_0)
    var_1 = dict_unpacking_transformer_0.visit_Dict(dict_0)
