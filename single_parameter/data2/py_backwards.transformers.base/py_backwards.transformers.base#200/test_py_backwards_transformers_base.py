# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.base as module_1


def test_case_0():
    yield_from_0 = module_0.YieldFrom()
    base_import_rewrite_0 = module_1.BaseImportRewrite(yield_from_0)


def test_case_1():
    index_0 = module_0.Index()
    list_0 = [index_0, index_0, index_0]
    import_from_0 = module_0.ImportFrom(*list_0)
    base_import_rewrite_0 = module_1.BaseImportRewrite(import_from_0)
    base_import_rewrite_0.visit_ImportFrom(import_from_0)


def test_case_2():
    index_0 = module_0.Index()
    none_type_0 = None
    list_0 = [none_type_0, none_type_0, none_type_0]
    dict_0 = {}
    import_from_0 = module_0.ImportFrom(*list_0, **dict_0)
    base_import_rewrite_0 = module_1.BaseImportRewrite(import_from_0)
    base_import_rewrite_0.visit_ImportFrom(import_from_0)
