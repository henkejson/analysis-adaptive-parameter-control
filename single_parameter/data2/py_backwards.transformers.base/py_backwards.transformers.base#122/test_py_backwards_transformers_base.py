# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.base as module_1
import typed_ast.ast3 as module_2


def test_case_0():
    import_0 = module_0.Import()
    base_node_transformer_0 = module_1.BaseNodeTransformer(import_0)


def test_case_1():
    continue_0 = module_0.Continue()
    base_import_rewrite_0 = module_1.BaseImportRewrite(continue_0)
    none_type_0 = None
    list_0 = [none_type_0, none_type_0, none_type_0]
    import_from_0 = module_0.ImportFrom(*list_0)
    base_import_rewrite_0.visit_ImportFrom(import_from_0)


def test_case_2():
    continue_0 = module_0.Continue()
    base_import_rewrite_0 = module_1.BaseImportRewrite(continue_0)
    list_0 = [base_import_rewrite_0, base_import_rewrite_0]
    import_from_0 = module_0.ImportFrom(*list_0)
    base_import_rewrite_0.visit_ImportFrom(import_from_0)


def test_case_3():
    none_type_0 = None
    list_0 = [none_type_0, none_type_0, none_type_0]
    import_from_0 = module_0.ImportFrom(*list_0)
    base_import_rewrite_0 = module_1.BaseImportRewrite(import_from_0)
    list_1 = [base_import_rewrite_0, list_0]
    import_from_1 = module_0.ImportFrom(*list_1)
    base_import_rewrite_0.visit_ImportFrom(import_from_1)


def test_case_4():
    continue_0 = module_0.Continue()
    base_import_rewrite_0 = module_1.BaseImportRewrite(continue_0)
    var_0 = module_2.iter_child_nodes(continue_0)
    list_0 = [var_0, var_0]
    import_from_0 = module_0.ImportFrom(*list_0)
    var_1 = base_import_rewrite_0.visit_ImportFrom(import_from_0)
