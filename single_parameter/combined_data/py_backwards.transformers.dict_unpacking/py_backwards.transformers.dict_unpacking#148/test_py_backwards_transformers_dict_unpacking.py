# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import py_backwards.transformers.dict_unpacking as module_0
import typed_ast._ast3 as module_1


def test_case_0():
    float_0 = -1484.0
    dict_unpacking_transformer_0 = module_0.DictUnpackingTransformer(float_0)


def test_case_1():
    none_type_0 = None
    str_0 = "\x0b*'*{"
    str_1 = "h"
    str_2 = "robotparser"
    str_3 = "F\\#0GSbUx{G_w-Z("
    dict_0 = {str_0: str_0, str_1: str_0, str_2: str_0, str_3: none_type_0}
    slice_0 = module_1.slice(**dict_0)
    dict_unpacking_transformer_0 = module_0.DictUnpackingTransformer(slice_0)
    dict_unpacking_transformer_0.visit_Module(none_type_0)
