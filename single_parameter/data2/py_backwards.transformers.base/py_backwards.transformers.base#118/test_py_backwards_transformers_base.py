# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast.ast3 as module_0
import py_backwards.transformers.base as module_1
import typed_ast._ast3 as module_2


def test_case_0():
    none_type_0 = None
    module_0.literal_eval(none_type_0)


def test_case_1():
    none_type_0 = None
    base_node_transformer_0 = module_1.BaseNodeTransformer(none_type_0)


def test_case_2():
    starred_0 = module_2.Starred()
    base_import_rewrite_0 = module_1.BaseImportRewrite(starred_0)
    list_0 = [starred_0]
    import_from_0 = module_2.ImportFrom(*list_0)
    base_import_rewrite_0.visit_ImportFrom(import_from_0)


def test_case_3():
    starred_0 = module_2.Starred()
    base_import_rewrite_0 = module_1.BaseImportRewrite(starred_0)
    base_node_transformer_0 = module_1.BaseNodeTransformer(starred_0)
    none_type_0 = None
    list_0 = [none_type_0, none_type_0]
    import_from_0 = module_2.ImportFrom(*list_0)
    base_import_rewrite_0.visit_ImportFrom(import_from_0)


def test_case_4():
    starred_0 = module_2.Starred()
    base_import_rewrite_0 = module_1.BaseImportRewrite(starred_0)
    list_0 = [base_import_rewrite_0, starred_0]
    list_1 = [list_0, list_0]
    import_from_0 = module_2.ImportFrom(*list_1)
    base_import_rewrite_0.visit_ImportFrom(import_from_0)


def test_case_5():
    starred_0 = module_2.Starred()
    base_import_rewrite_0 = module_1.BaseImportRewrite(starred_0)
    var_0 = module_0.iter_child_nodes(starred_0)
    list_0 = [var_0, var_0]
    import_from_0 = module_2.ImportFrom(*list_0)
    var_1 = base_import_rewrite_0.visit_ImportFrom(import_from_0)
