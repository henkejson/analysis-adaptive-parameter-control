# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast.ast3 as module_0
import typed_ast._ast3 as module_1
import py_backwards.transformers.dict_unpacking as module_2


def test_case_0():
    none_type_0 = None
    module_0.increment_lineno(none_type_0, none_type_0)


def test_case_1():
    stmt_0 = module_1.stmt()
    dict_unpacking_transformer_0 = module_2.DictUnpackingTransformer(stmt_0)
    dict_unpacking_transformer_1 = module_2.DictUnpackingTransformer(stmt_0)
    dict_unpacking_transformer_1.visit_Module(dict_unpacking_transformer_0)
