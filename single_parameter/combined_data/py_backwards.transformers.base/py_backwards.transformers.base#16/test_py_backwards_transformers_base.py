# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.base as module_1


def test_case_0():
    expr_0 = module_0.expr()
    base_node_transformer_0 = module_1.BaseNodeTransformer(expr_0)


def test_case_1():
    stmt_0 = module_0.stmt()
    list_0 = [stmt_0]
    import_from_0 = module_0.ImportFrom(*list_0)
    base_import_rewrite_0 = module_1.BaseImportRewrite(stmt_0)
    base_import_rewrite_0.visit_ImportFrom(import_from_0)


def test_case_2():
    stmt_0 = module_0.stmt()
    list_0 = [stmt_0, stmt_0, stmt_0, stmt_0]
    list_1 = [stmt_0, list_0, stmt_0]
    import_from_0 = module_0.ImportFrom(*list_1)
    base_import_rewrite_0 = module_1.BaseImportRewrite(list_1)
    base_import_rewrite_0.visit(import_from_0)


def test_case_3():
    list_0 = []
    list_1 = [list_0, list_0, list_0]
    import_from_0 = module_0.ImportFrom(*list_1)
    base_import_rewrite_0 = module_1.BaseImportRewrite(list_1)
    var_0 = base_import_rewrite_0.visit_ImportFrom(import_from_0)
