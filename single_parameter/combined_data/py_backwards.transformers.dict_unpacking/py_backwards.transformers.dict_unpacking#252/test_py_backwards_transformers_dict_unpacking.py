# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.dict_unpacking as module_1


def test_case_0():
    slice_0 = module_0.slice()
    dict_unpacking_transformer_0 = module_1.DictUnpackingTransformer(slice_0)


def test_case_1():
    mod_0 = module_0.Mod()
    dict_unpacking_transformer_0 = module_1.DictUnpackingTransformer(mod_0)
    r_shift_0 = module_0.RShift()
    dict_unpacking_transformer_1 = module_1.DictUnpackingTransformer(r_shift_0)
    dict_unpacking_transformer_1.visit_Module(r_shift_0)
