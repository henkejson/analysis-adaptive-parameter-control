# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast.ast3 as module_0
import typed_ast._ast3 as module_1
import py_backwards.transformers.dict_unpacking as module_2


def test_case_0():
    none_type_0 = None
    module_0.fix_missing_locations(none_type_0)


def test_case_1():
    add_0 = module_1.Add()
    dict_unpacking_transformer_0 = module_2.DictUnpackingTransformer(add_0)
    var_0 = module_0.iter_child_nodes(add_0)
    list_0 = [var_0, var_0]
    dict_0 = module_1.Dict(*list_0)
    var_1 = dict_unpacking_transformer_0.visit_Dict(dict_0)
    dict_unpacking_transformer_0.visit_Module(var_0)


def test_case_2():
    add_0 = module_1.Add()
    dict_unpacking_transformer_0 = module_2.DictUnpackingTransformer(add_0)
    var_0 = module_0.iter_child_nodes(add_0)
    list_0 = [var_0, var_0]
    dict_0 = module_1.Dict(*list_0)
    var_1 = dict_unpacking_transformer_0.visit_Dict(dict_0)
