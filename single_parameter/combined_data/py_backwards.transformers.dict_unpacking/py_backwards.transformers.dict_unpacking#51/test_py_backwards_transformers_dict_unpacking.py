# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.dict_unpacking as module_1


def test_case_0():
    floor_div_0 = module_0.FloorDiv()
    dict_unpacking_transformer_0 = module_1.DictUnpackingTransformer(floor_div_0)


def test_case_1():
    for_0 = module_0.For()
    dict_unpacking_transformer_0 = module_1.DictUnpackingTransformer(for_0)
    dict_unpacking_transformer_0.visit_Module(dict_unpacking_transformer_0)
