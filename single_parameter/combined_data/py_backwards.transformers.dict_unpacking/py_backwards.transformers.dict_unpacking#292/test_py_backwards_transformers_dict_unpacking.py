# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import py_backwards.transformers.dict_unpacking as module_0
import typed_ast._ast3 as module_1
import typed_ast.ast3 as module_2


def test_case_0():
    int_0 = -2485
    dict_unpacking_transformer_0 = module_0.DictUnpackingTransformer(int_0)


def test_case_1():
    str_0 = "#Vy!t[n,[Jb+\x0bAT|"
    str_1 = "h)NZf0&M.m%O42"
    dict_0 = {str_0: str_0, str_1: str_0}
    u_add_0 = module_1.UAdd(**dict_0)
    dict_unpacking_transformer_0 = module_0.DictUnpackingTransformer(u_add_0)
    stmt_0 = module_1.stmt()
    dict_unpacking_transformer_1 = module_0.DictUnpackingTransformer(stmt_0)
    var_0 = module_2.iter_child_nodes(str_1)
    dict_unpacking_transformer_0.visit_Module(var_0)
