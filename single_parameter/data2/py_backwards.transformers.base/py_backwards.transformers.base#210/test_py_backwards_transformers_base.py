# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.base as module_1


def test_case_0():
    list_0 = module_0.List()
    base_import_rewrite_0 = module_1.BaseImportRewrite(list_0)


def test_case_1():
    none_type_0 = None
    list_0 = [none_type_0, none_type_0, none_type_0]
    import_from_0 = module_0.ImportFrom(*list_0)
    base_import_rewrite_0 = module_1.BaseImportRewrite(import_from_0)
    base_import_rewrite_0.visit_ImportFrom(import_from_0)


def test_case_2():
    str_0 = ""
    base_import_rewrite_0 = module_1.BaseImportRewrite(str_0)
    list_0 = [base_import_rewrite_0, str_0, str_0]
    import_from_0 = module_0.ImportFrom(*list_0)
    var_0 = base_import_rewrite_0.visit_ImportFrom(import_from_0)


def test_case_3():
    str_0 = "t"
    base_import_rewrite_0 = module_1.BaseImportRewrite(str_0)
    list_0 = [base_import_rewrite_0, str_0, str_0]
    import_from_0 = module_0.ImportFrom(*list_0)
    base_import_rewrite_0.visit_ImportFrom(import_from_0)
