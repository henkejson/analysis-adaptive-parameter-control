# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.base as module_1
import typed_ast.ast3 as module_2


def test_case_0():
    with_0 = module_0.With()
    base_node_transformer_0 = module_1.BaseNodeTransformer(with_0)


def test_case_1():
    none_type_0 = None
    list_0 = [none_type_0, none_type_0]
    import_from_0 = module_0.ImportFrom(*list_0)
    not_eq_0 = module_0.NotEq()
    base_import_rewrite_0 = module_1.BaseImportRewrite(not_eq_0)
    base_import_rewrite_0.visit_ImportFrom(import_from_0)


def test_case_2():
    dict_0 = {}
    list_0 = [dict_0, dict_0]
    import_from_0 = module_0.ImportFrom(*list_0)
    base_import_rewrite_0 = module_1.BaseImportRewrite(list_0)
    var_0 = base_import_rewrite_0.visit_ImportFrom(import_from_0)


def test_case_3():
    dict_0 = {}
    list_0 = [dict_0, dict_0]
    list_1 = [list_0, list_0, list_0]
    import_from_0 = module_0.ImportFrom(*list_1, **dict_0)
    var_0 = module_2.walk(import_from_0)
    base_import_rewrite_0 = module_1.BaseImportRewrite(var_0)
    base_import_rewrite_0.visit_ImportFrom(import_from_0)
