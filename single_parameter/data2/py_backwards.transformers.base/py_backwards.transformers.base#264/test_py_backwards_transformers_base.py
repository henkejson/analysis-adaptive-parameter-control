# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.base as module_1


def test_case_0():
    store_0 = module_0.Store()
    base_import_rewrite_0 = module_1.BaseImportRewrite(store_0)


def test_case_1():
    yield_0 = module_0.Yield()
    base_import_rewrite_0 = module_1.BaseImportRewrite(yield_0)
    none_type_0 = None
    list_0 = [none_type_0]
    import_from_0 = module_0.ImportFrom(*list_0)
    base_import_rewrite_0.visit_ImportFrom(import_from_0)


def test_case_2():
    yield_0 = module_0.Yield()
    base_import_rewrite_0 = module_1.BaseImportRewrite(yield_0)
    list_0 = [base_import_rewrite_0]
    import_from_0 = module_0.ImportFrom(*list_0)
    base_import_rewrite_0.visit_ImportFrom(import_from_0)


def test_case_3():
    yield_0 = module_0.Yield()
    base_import_rewrite_0 = module_1.BaseImportRewrite(yield_0)
    list_0 = [base_import_rewrite_0]
    list_1 = [base_import_rewrite_0, list_0, yield_0]
    import_from_0 = module_0.ImportFrom(*list_1)
    base_import_rewrite_0.visit_ImportFrom(import_from_0)


def test_case_4():
    yield_0 = module_0.Yield()
    base_import_rewrite_0 = module_1.BaseImportRewrite(yield_0)
    list_0 = []
    list_1 = [base_import_rewrite_0, list_0, yield_0]
    import_from_0 = module_0.ImportFrom(*list_1)
    var_0 = base_import_rewrite_0.visit_ImportFrom(import_from_0)
