# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.dict_unpacking as module_1
import typed_ast.ast3 as module_2


def test_case_0():
    async_for_0 = module_0.AsyncFor()
    dict_unpacking_transformer_0 = module_1.DictUnpackingTransformer(async_for_0)


def test_case_1():
    aug_assign_0 = module_0.AugAssign()
    dict_unpacking_transformer_0 = module_1.DictUnpackingTransformer(aug_assign_0)
    none_type_0 = None
    var_0 = module_2.walk(none_type_0)
    dict_unpacking_transformer_0.visit_Module(var_0)
