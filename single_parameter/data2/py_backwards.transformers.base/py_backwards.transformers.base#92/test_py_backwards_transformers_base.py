# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import py_backwards.transformers.base as module_0
import typed_ast._ast3 as module_1
import typed_ast.ast3 as module_2


def test_case_0():
    dict_0 = {}
    base_node_transformer_0 = module_0.BaseNodeTransformer(dict_0)


def test_case_1():
    load_0 = module_1.Load()
    base_import_rewrite_0 = module_0.BaseImportRewrite(load_0)
    list_0 = [base_import_rewrite_0, base_import_rewrite_0]
    import_from_0 = module_1.ImportFrom(*list_0)
    base_import_rewrite_0.visit_ImportFrom(import_from_0)


def test_case_2():
    load_0 = module_1.Load()
    var_0 = module_2.iter_fields(load_0)
    base_import_rewrite_0 = module_0.BaseImportRewrite(load_0)
    list_0 = [base_import_rewrite_0, var_0]
    import_from_0 = module_1.ImportFrom(*list_0)
    var_1 = base_import_rewrite_0.visit_ImportFrom(import_from_0)


def test_case_3():
    load_0 = module_1.Load()
    var_0 = module_2.walk(load_0)
    base_import_rewrite_0 = module_0.BaseImportRewrite(load_0)
    list_0 = [base_import_rewrite_0, var_0]
    import_from_0 = module_1.ImportFrom(*list_0)
    base_import_rewrite_0.visit_ImportFrom(import_from_0)
